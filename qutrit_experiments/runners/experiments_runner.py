"""Standard experiments driver."""

import logging
import os
import json
import lzma
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime
from numbers import Number
from typing import Optional, Union
import matplotlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives.containers import BitArray
from qiskit.providers import Backend, JobStatus
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.framework import (AnalysisResultTable, AnalysisStatus, BaseExperiment,
                                          CompositeAnalysis as CompositeAnalysisOrig,
                                          ExperimentData, ExperimentDecoder, ExperimentEncoder)
from qiskit_ibm_runtime import Batch, SamplerV2 as Sampler
from qiskit_ibm_runtime.base_runtime_job import BaseRuntimeJob
from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError, IBMRuntimeError
from qiskit_ibm_runtime.utils.default_session import get_cm_session

from ..constants import DEFAULT_REP_DELAY, DEFAULT_SHOTS, RESTLESS_REP_DELAY
from ..experiment_config import (CompositeExperimentConfig, ExperimentConfigBase,
                                 experiments, postexperiments)
from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework_overrides.composite_analysis import CompositeAnalysis
from ..framework_overrides.parallel_experiment import ParallelExperiment
from ..transpilation.qutrit_transpiler import (BASIS_GATES, QutritTranspileOptions,
                                               make_instruction_durations,
                                               transpile_qutrit_circuits)


def display(_):  # pylint: disable=missing-function-docstring
    pass


if 'backend_inline' in matplotlib.rcParams['backend']:
    try:
        from IPython.display import display  # noqa: F811
    except ImportError:
        pass

LOG = logging.getLogger(__name__)


class ExperimentsRunner:
    """Experiments driver.

    ExperimentsRunner provides a common environment to run the experiments, manage the result data,
    and update the calibrations. It is responsible for constructing the experiment instance from an
    ExperimentConfig object, circuit transpilation accounting for qutrit-specific requirements,
    submitting circuit jobs, saving / recalling the experiment data to / from disk, and updating
    the calibrations.
    """
    def __init__(
        self,
        backend: Backend,
        qubits: Optional[Sequence[int]] = None,
        calibrations: Optional[Union[Calibrations, str]] = None,
        data_dir: Optional[str] = None,
        read_only: bool = False
    ):
        self.backend = backend
        self.qubits = qubits
        self.data_dir = data_dir

        if isinstance(calibrations, str):
            self.load_calibrations(file_name=calibrations)
        else:
            self.calibrations = calibrations

        self.read_only = read_only
        self.default_print_level = 2

        self._skip_missing_calibration = False

        self.job_retry_interval = -1.

        self.qutrit_transpile_options = QutritTranspileOptions()

        self.program_data = {}

    @property
    def qubits(self):
        return self._qubits

    @qubits.setter
    def qubits(self, value: Union[Sequence[int], None]):
        self._qubits = None if value is None else tuple(value)

    @property
    def data_dir(self) -> str:
        return self._data_dir

    @data_dir.setter
    def data_dir(self, path: Union[str, None]):
        if path and not os.path.exists(path):
            os.makedirs(path)
        if not os.path.isdir(pdata_path := os.path.join(path, 'program_data')):
            os.mkdir(pdata_path)
        self._data_dir = path

    def load_calibrations(
        self,
        file_name: str = 'calibrations.json',
        data_dir: Optional[str] = None,
        is_csv: bool = False  # for backward compatibility
    ):
        data_dir = data_dir or self._data_dir
        if is_csv:
            self.calibrations.load_parameter_values(file_name=os.path.join(data_dir, file_name))
        else:
            self.calibrations = Calibrations.load(os.path.join(data_dir, file_name))

    def make_experiment(
        self,
        config: Union[str, ExperimentConfigBase]
    ) -> BaseExperiment:
        return self._make_experiment(config)

    def _make_experiment(
        self,
        config: Union[str, ExperimentConfigBase]
    ) -> BaseExperiment:
        """Indirection to avoid infinite loops when making CompositeExperiments from a subclass."""
        if isinstance(config, str):
            config = experiments[config](self)

        args = {'backend': self.backend}
        if isinstance(config, CompositeExperimentConfig):
            args['experiments'] = [self._make_experiment(sub) for sub in config.subexperiments]
            args['flatten_results'] = config.flatten_results
        else:
            args.update(config.args)
            if config.physical_qubits is not None:
                args['physical_qubits'] = config.physical_qubits
            if issubclass(config.cls, BaseCalibrationExperiment):
                args['calibrations'] = self.calibrations

        experiment = config.cls(**args)
        experiment.experiment_type = config.exp_type
        if not config.analysis:
            experiment.analysis = None

        experiment.set_experiment_options(**config.experiment_options)
        if isinstance(config, CompositeExperimentConfig):
            # CompositeExperiment creates a CompositeAnalysisOrig by default; overwrite to the
            # serial version
            # pylint: disable-next=unidiomatic-typecheck
            if type(experiment.analysis) is CompositeAnalysisOrig:
                analyses = experiment.analysis.component_analysis()
                flatten_results = experiment.analysis._flatten_results
                experiment.analysis = CompositeAnalysis(analyses, flatten_results=flatten_results)
        else:
            if isinstance(experiment, BaseCalibrationExperiment):
                experiment.auto_update = False
            if config.restless:
                experiment.enable_restless(rep_delay=RESTLESS_REP_DELAY)

        if experiment.analysis is not None:
            experiment.analysis.set_options(**config.analysis_options)

        return experiment

    def run_experiment(
        self,
        config: Union[str, ExperimentConfigBase],
        experiment: Optional[BaseExperiment] = None,
        block_for_results: bool = True,
        analyze: bool = True,
        calibrate: bool = True,
        print_level: Optional[int] = None,
        exp_data: Optional[ExperimentData] = None,
        force_resubmit: bool = False
    ) -> ExperimentData:
        if isinstance(config, str):
            config = experiments[config](self)
        if experiment is None:
            experiment = self.make_experiment(config)
        exp_type = experiment.experiment_type

        LOG.info('run_experiment(%s)', exp_type)

        experiment.set_run_options(**config.run_options)
        if experiment.analysis is None:
            analyze = False

        if force_resubmit:
            self.delete_data(exp_type)

        def exec_or_add(task):
            """Execute the task directly or add it as an analysis callback. Direct execution makes
            for easier debugging."""
            if block_for_results:
                # Wait here once just to make the program execution easier to understand
                self._check_status(exp_data)
                task(exp_data)
            else:
                exp_data.add_analysis_callback(task)

        if exp_data is None:
            # Try retrieving exp_data from json
            if self.saved_data_exists(exp_type):
                exp_data = self.load_data(exp_type)
            else:
                # Construct the data from running or retrieving jobs -> publish the raw data
                # Finalize the experiment before executions
                experiment._finalize()
                # Initialize the result container
                # This line comes after _transpiled_circuits in the original BaseExperiment.run()
                # (see below) but in terms of good object design there shouldn't be any dependencies
                # between the two methods
                exp_data = experiment._initialize_experiment_data()

                if self.job_ids_exist(exp_type):
                    jobs = self.load_jobs(exp_type)
                else:
                    jobs = self._submit_jobs(experiment)

                for job in jobs:
                    jid = job.job_id()
                    exp_data.job_ids.append(jid)
                    exp_data._jobs[jid] = job
                    if block_for_results:
                        self._add_job_data(exp_data, job)
                    else:
                        exp_data._job_futures[jid] = exp_data._job_executor.submit(
                            self._add_job_data, exp_data, job
                        )

                exec_or_add(self.save_data)

        if not analyze:
            LOG.info('No analysis will be performed for %s.', exp_type)
            return exp_data

        if block_for_results:
            LOG.info('Running the analysis for %s', exp_type)
        else:
            LOG.info('Reserving the analysis for %s', exp_type)
        experiment.analysis.run(exp_data)

        if block_for_results:
            self._check_status(exp_data)

        if calibrate and self.calibrations is not None:
            def update_calibrations(exp_data):
                self.update_calibrations(exp_data, experiment=experiment,
                                         criterion=config.calibration_criterion)

            exec_or_add(update_calibrations)

        if (postexp := postexperiments.get(exp_type)) is not None:
            def postexperiment(exp_data):
                LOG.info('Performing the postexperiment for %s.', exp_type)
                postexp(self, exp_data)

            exec_or_add(postexperiment)

        if block_for_results:
            if print_level is None:
                print_level = self.default_print_level

            self._check_status(exp_data)
            if print_level == 1:
                print_summary(exp_data)
            elif print_level == 2:
                print_details(exp_data)

        return exp_data

    def get_transpiled_circuits(
        self,
        experiment: Union[BaseExperiment, ExperimentConfigBase],
        transpiled_circuits: Optional[list[QuantumCircuit]] = None
    ) -> list[QuantumCircuit]:
        """Return a list of transpiled circuits accounting for qutrit-specific instructions."""
        if isinstance(experiment, str):
            experiment = experiments[experiment](self)
        if isinstance(experiment, ExperimentConfigBase):
            experiment = self.make_experiment(experiment)

        if self.calibrations is None:
            # Nothing to do
            return transpiled_circuits or experiment._transpiled_circuits()

        instruction_durations = make_instruction_durations(self.backend, self.calibrations,
                                                           qubits=experiment.physical_qubits)

        if not transpiled_circuits:
            if not isinstance(experiment, BaseCalibrationExperiment):
                experiment.set_transpile_options(
                    # By setting the basis_gates, PassManagerConfig.from_backend() will not take the
                    # target from the backend, making target absent everywhere in the preset pass
                    # manager. When the target is None, HighLevelSynthesis (responsible for
                    # translating all gates to basis gates) will reference the passed basis_gates
                    # list and leaves all gates appearing in the list untouched.
                    basis_gates=self.backend.basis_gates + [g.gate_name for g in BASIS_GATES],
                    # Scheduling method has to be specified in case there are delay instructions
                    # that violate the alignment constraints, in which case a
                    # ConstrainedRescheduling is triggered, which fails without precalculated
                    # node_start_times.
                    scheduling_method='alap',
                    # And to run the scheduler, durations of all gates must be known.
                    instruction_durations=instruction_durations
                )

            start = time.time()
            transpiled_circuits = experiment._transpiled_circuits()
            end = time.time()
            LOG.debug('Initial transpilation took %.1f seconds.', end - start)

        start = time.time()
        transpiled_circuits = transpile_qutrit_circuits(transpiled_circuits,
                                                        self.backend, self.calibrations,
                                                        qubit_transpiled=True,
                                                        instruction_durations=instruction_durations,
                                                        options=self.qutrit_transpile_options)
        end = time.time()
        LOG.debug('Qutrit-specific transpilation took %.1f seconds.', end - start)
        return transpiled_circuits

    def save_data(
        self,
        experiment_data: ExperimentData
    ):
        if not self._data_dir:
            return
        # _experiment and backend attributes may contain unserializable objects
        experiment = experiment_data.experiment
        backend = experiment_data.backend
        experiment_data._experiment = None
        experiment_data.backend = None
        copy_data = experiment_data.copy(copy_results=False)
        experiment_data._experiment = experiment
        experiment_data.backend = backend

        json_bytes = json.dumps(copy_data, cls=ExperimentEncoder).encode('utf-8')
        file_name = os.path.join(self._data_dir, f'{experiment_data.experiment_type}.json.xz')
        with lzma.open(file_name, 'w') as out:
            out.write(json_bytes)

    def saved_data_exists(self, config: Union[str, ExperimentConfigBase]) -> bool:
        if isinstance(config, str):
            exp_type = config
        else:
            exp_type = config.exp_type
        return self._file_exists(f'{exp_type}.json.xz')

    def delete_data(self, exp_type: str):
        for suffix in ['_jobs.dat', '_failedjobs.dat', '.json.xz']:
            try:
                os.unlink(os.path.join(self._data_dir, f'{exp_type}{suffix}'))
            except FileNotFoundError:
                continue

    def job_ids_exist(self, exp_type: str) -> bool:
        return self._file_exists(f'{exp_type}_jobs.dat')

    def _file_exists(self, file_name: str) -> bool:
        if not self._data_dir:
            return False
        return os.path.exists(os.path.join(self._data_dir, file_name))

    def load_data(self, config: Union[str, ExperimentConfigBase]) -> ExperimentData:
        if not self._data_dir:
            raise RuntimeError('ExperimentsRunner does not have a data_dir set')

        if isinstance(config, str):
            exp_type = config
        else:
            exp_type = config.exp_type

        start = time.time()
        file_name = os.path.join(self._data_dir, f'{exp_type}.json.xz')
        with lzma.open(file_name, 'r') as source:
            exp_data = json.loads(source.read(), cls=ExperimentDecoder)

        # Bugfix (QE 0.6) JSON-(de)serialization of empty AnalysisResultTable does not close
        if exp_data._analysis_results._data.size == 0:
            exp_data._analysis_results = AnalysisResultTable()

        end = time.time()

        LOG.info('Unpacked experiment data for %s.', exp_type)
        LOG.debug('Loading data required %.1f seconds.', end - start)
        return exp_data

    def load_jobs(self, exp_type: str) -> list[BaseRuntimeJob]:
        if not self._data_dir:
            raise RuntimeError('ExperimentsRunner does not have a data_dir set')
        with open(os.path.join(self._data_dir, f'{exp_type}_jobs.dat'), encoding='utf-8') as source:
            num_jobs = len(source.readline().strip().split())
            job_unique_tag = source.readline().strip()

        LOG.info('Retrieving runtime job with unique id %s', job_unique_tag)
        while True:
            try:
                return self.backend.service.jobs(limit=num_jobs, backend_name=self.backend.name,
                                                 job_tags=[job_unique_tag])
            except IBMNotAuthorizedError:
                continue

    def update_calibrations(
        self,
        experiment_data: ExperimentData,
        experiment: Optional[BaseExperiment] = None,
        exp_type: Optional[str] = None,
        criterion: Optional[Callable[[ExperimentData], bool]] = None
    ):
        if experiment is None:
            experiment = experiment_data.experiment
        if exp_type:
            LOG.info('Updating calibrations for %s.', exp_type)

        def _get_update_list(exp_data, exp):
            update_list = []
            if isinstance(exp, BaseCalibrationExperiment):
                if exp_type:
                    etype = exp_type
                else:
                    etype = exp.experiment_type
                    LOG.info('Updating calibrations for %s.', etype)

                updated = False
                try:
                    if criterion and not criterion(exp_data):
                        LOG.warning('%s qubits %s failed calibration criterion', etype,
                                    exp.physical_qubits)
                    else:
                        exp.update_calibrations(exp_data)
                        updated = True
                except ExperimentEntryNotFound as exc:
                    if self._skip_missing_calibration:
                        LOG.warning('%s (%s) %s', etype, exp.physical_qubits, exc.message)
                    else:
                        raise

                param_name = exp._param_name
                sched_name = exp._sched_name
                if isinstance(param_name, str):
                    targets = [(param_name, sched_name)]
                elif isinstance(sched_name, str):
                    targets = [(pname, sched_name) for pname in param_name]
                else:
                    targets = list(zip(param_name, sched_name))
                qubits_map = exp.experiment_options.get('calibration_qubit_index', {})
                for pname, sname in targets:
                    qubit_indices = qubits_map.get((pname, sname), range(len(exp.physical_qubits)))
                    qubits = tuple(exp.physical_qubits[idx] for idx in qubit_indices)
                    update_list.append((pname, sname, etype, qubits, updated))

            elif type(exp) in [BatchExperiment, ParallelExperiment]:
                for subexp, child_data in zip(exp.component_experiment(), exp_data.child_data()):
                    update_list.extend(_get_update_list(child_data, subexp))

            return update_list

        update_list = _get_update_list(experiment_data, experiment)
        LOG.info('%d/%d parameters to update.',
                 len([x for x in update_list if x[-1]]), len(update_list))

        for pname, sname, etype, qubits, updated in update_list:
            if not updated:
                continue

            LOG.debug('Tagging calibration parameter %s:%s:%s from experiment %s',
                      pname, sname, qubits, etype)
            self.pass_parameter_value(pname, qubits, from_schedule=sname, from_group='default',
                                      to_group=etype)

        if any(x[-1] for x in update_list) and self._data_dir and not self.read_only:
            self.calibrations.save(folder=self._data_dir, overwrite=True,
                                   file_prefix='calibrations')

    def save_program_data(self, key: Optional[str] = None):
        if not self._data_dir:
            return
        pdata_path = os.path.join(self._data_dir, 'program_data')
        if key is None:
            keys = self.program_data.keys()
        elif isinstance(key, str):
            keys = [key]
        else:
            keys = key
        for key in keys:
            json_bytes = json.dumps(self.program_data[key], cls=ExperimentEncoder).encode('utf-8')
            file_name = os.path.join(pdata_path, f'{key}.json.xz')
            with lzma.open(file_name, 'w') as out:
                out.write(json_bytes)

    def load_program_data(
        self,
        keys: Optional[Union[str, list[str]]] = None,
        allow_missing: bool = False
    ):
        if (not self._data_dir
                or not os.path.isdir(pdata_path := os.path.join(self._data_dir, 'program_data'))):
            raise RuntimeError('Program data directory does not exist')
        if not keys:
            keys = map(lambda s: s.replace('.json.xz', ''), os.listdir(pdata_path))
        elif isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                file_name = os.path.join(pdata_path, f'{key}.json.xz')
                with lzma.open(file_name, 'r') as source:
                    self.program_data[key] = json.loads(source.read(), cls=ExperimentDecoder)
            except FileNotFoundError:
                if allow_missing:
                    LOG.info('Program data %s not found.', key)
                else:
                    raise

    def _submit_jobs(
        self,
        experiment: BaseExperiment,
    ) -> ExperimentData:
        LOG.info('Creating transpiled circuits for %s.', experiment.experiment_type)
        # Generate and transpile circuits
        start = time.time()
        transpiled_circuits = self.get_transpiled_circuits(experiment)
        end = time.time()
        LOG.debug('Created %d circuits for %s in %.1f seconds.',
                  len(transpiled_circuits), experiment.experiment_type, end - start)

        # Add a tag to the job to make later identification easier
        job_unique_tag = uuid.uuid4().hex

        # Sampler options
        sampler_options = {
            'default_shots': DEFAULT_SHOTS,
            'execution': {'rep_delay': DEFAULT_REP_DELAY},
            'environment': {
                'job_tags': [experiment.experiment_type, job_unique_tag]
            }
        }
        run_opts = experiment.run_options
        if (init := run_opts.get('init_qubits')) is not None:
            sampler_options['execution']['init_qubits'] = init
        if (delay := run_opts.get('rep_delay')) is not None:
            sampler_options['execution']['rep_delay'] = delay
        match run_opts.get('meas_level'):
            case MeasLevel.CLASSIFIED | 2:
                sampler_options['execution']['meas_type'] = 'classified'
            case MeasLevel.KERNELED | 1:
                match run_opts.get('meas_return'):
                    case MeasReturnType.SINGLE | 'single':
                        sampler_options['execution']['meas_type'] = 'kerneled'
                    case _:
                        LOG.warning('Run option meas_return="avg" is set. Information on number of'
                                    ' shots will be lost from the results.')
                        sampler_options['execution']['meas_type'] = 'avg_kerneled'

        # Run jobs (reimplementing BaseExperiment._run_jobs to use Sampler & get a handle on job
        # splitting)
        circuit_lists = self._split_circuits(experiment, transpiled_circuits)
        LOG.debug('Circuits will be split into %d jobs.', len(circuit_lists))

        jobs = []

        LOG.info('Submitting experiment circuit jobs for %s.', experiment.experiment_type)

        # Detect a session context. If none and len(jobs) > 1, create a batch
        if (session := get_cm_session()) is None and len(circuit_lists) > 1:
            session = Batch(backend=self.backend)

        try:
            # Session takes precedence
            sampler = Sampler(backend=self.backend, session=session, options=sampler_options)
            # Check all jobs got an id. If not, try resubmitting
            for ilist, circs in enumerate(circuit_lists):
                for _ in range(5):
                    while True:
                        try:
                            job = sampler.run(circs, shots=run_opts.get('shots'))
                            break
                        except (IBMRuntimeError, IBMNotAuthorizedError) as ex:
                            if self.job_retry_interval < 0.:
                                raise
                            LOG.error('IBMRuntimeError during job submission: %s',  ex.message)

                        time.sleep(self.job_retry_interval)

                    if job.job_id():
                        jobs.append(job)
                        break
                else:
                    start = sum(len(lst) for lst in circuit_lists[:ilist])
                    end = start + len(circs) - 1
                    raise RuntimeError(f'Failed to submit circuits {start}-{end}')
        finally:
            if get_cm_session() is None and session is not None:
                session.close()

        LOG.info('Job IDs: %s', [job.job_id() for job in jobs])

        if self._data_dir:
            job_ids_path = os.path.join(self._data_dir, f'{experiment.experiment_type}_jobs.dat')
            with open(job_ids_path, 'w', encoding='utf-8') as out:
                out.write(' '.join(job.job_id() for job in jobs) + '\n')
                out.write(job_unique_tag + '\n')

        return jobs

    def _add_job_data(self, experiment_data, job):
        jid = job.job_id()
        try:
            job_result = job.result()
            experiment_data._running_time = None

            with experiment_data._result_data.lock:
                # Lock data while adding all result data
                for pub_result in job_result:
                    # pub_result.data is a dict-like object with elements corresponding to cregs
                    creg_data = list(pub_result.data.values())
                    if isinstance(creg_data[0], BitArray):
                        # meas_type == 'classified'
                        if len(creg_data) == 1:
                            result_data = creg_data[0]
                        else:
                            # More than one cregs -> Concatenate packed bits
                            bitstrings_list = [c.get_bitstrings() for c in creg_data]
                            result_data = BitArray.from_samples(
                                [''.join(bitstrings[::-1]) for bitstrings in zip(*bitstrings_list)]
                            )

                        data = {'counts': result_data.get_counts()}
                        data['meas_level'] = 2
                        data['shots'] = result_data.num_shots
                    else:
                        # meas_type == 'avg_kerneled' or 'kerneled'
                        if len(creg_data) == 1:
                            result_data = creg_data[0]
                        else:
                            # More than one cregs -> Concatenate the memory arrays
                            result_data = np.concatenate(creg_data, axis=-1)

                        memory = np.stack([result_data.real, result_data.imag], axis=-1).tolist()
                        data = {'memory': memory}
                        data['meas_level'] = 1
                        if result_data.ndim == 1:
                            data['meas_return'] = 'avg'
                            # This is incorrect but is the only thing we can do absent actually
                            # querying the job inputs
                            data['shots'] = 1
                        else:
                            data['meas_return'] = 'single'
                            data['shots'] = result_data.shape[0]

                    data['job_id'] = jid
                    data['metadata'] = pub_result.metadata['circuit_metadata']
                    experiment_data._result_data.append(data)

            LOG.debug("Job data added [Job ID: %s]", jid)
            # sets the endtime to be the time the last successful job was added
            experiment_data.end_datetime = datetime.now()
            return jid, True
        except Exception as ex:  # pylint: disable=broad-except
            if self._data_dir:
                job_ids_path = os.path.join(self._data_dir,
                                            f'{experiment_data.experiment_type}_jobs.dat')
                failed_jobs_path = os.path.join(self._data_dir,
                                                f'{experiment_data.experiment_type}_failedjobs.dat')
                with open(failed_jobs_path, 'a', encoding='utf-8') as out:
                    with open(job_ids_path, 'r', encoding='utf-8') as source:
                        out.write(source.read())

                os.unlink(job_ids_path)

            # Handle cancelled jobs
            status = job.status()
            if status == JobStatus.CANCELLED:
                LOG.warning("Job was cancelled before completion [Job ID: %s]", jid)
                return jid, False
            if status == JobStatus.ERROR:
                LOG.error(
                    "Job data not added for errored job [Job ID: %s]\nError message: %s",
                    jid,
                    job.error_message(),
                )
                return jid, False
            LOG.warning("Adding data from job failed [Job ID: %s]", job.job_id())
            raise ex

    def _check_status(self, experiment_data: ExperimentData):
        LOG.debug('Checking the status of %s', experiment_data.experiment_type)
        experiment_data.block_for_results()
        if (status := experiment_data.analysis_status()) != AnalysisStatus.DONE:
            raise RuntimeError(f'Post-job status = {status.value}')

    def pass_parameter_value(
        self,
        from_param: str,
        from_qubits: Union[int, tuple[int, ...]],
        from_schedule: Optional[str] = None,
        from_group: str = 'default',
        to_param: Optional[str] = None,
        to_qubits: Optional[Union[int, tuple[int, ...]]] = None,
        to_schedule: Optional[str] = None,
        to_group: Optional[str] = None,
        transform: Optional[Callable[[Number], Number]] = None
    ):
        if isinstance(from_qubits, int):
            from_qubits = (from_qubits,)

        param_key = (from_param, from_qubits, from_schedule)
        parameter_values = [pv for pv in self.calibrations._params[param_key]
                            if pv.group == from_group]
        from_value = max(enumerate(parameter_values), key=lambda x: (x[1].date_time, x[0]))[1]

        if to_param is None:
            to_param = from_param
        if to_qubits is None:
            to_qubits = from_qubits
        if to_schedule is None:
            to_schedule = from_schedule
        if to_group is None:
            to_group = from_group

        value = from_value.value
        if transform is not None:
            value = transform(value)

        LOG.info('Adding parameter value %s for %s qubits=%s schedule=%s',
                 value, to_param, to_qubits, to_schedule)

        to_value = ParameterValue(
            value=value,
            date_time=from_value.date_time,
            valid=from_value.valid,
            exp_id=from_value.exp_id,
            group=to_group
        )
        self.calibrations.add_parameter_value(to_value, to_param, qubits=to_qubits,
                                              schedule=to_schedule)

    @staticmethod
    def _split_circuits(experiment, circuits):
        max_circuits_option = getattr(experiment.experiment_options, "max_circuits", None)
        max_circuits_backend = experiment._backend_data.max_circuits
        if max_circuits_option and max_circuits_backend:
            max_circuits = min(max_circuits_option, max_circuits_backend)
        elif max_circuits_option:
            max_circuits = max_circuits_option
        else:
            max_circuits = max_circuits_backend

        # Run experiment jobs
        if max_circuits and len(circuits) > max_circuits:
            # Split jobs for backends that have a maximum job size
            job_circuits = [
                circuits[i:i + max_circuits] for i in range(0, len(circuits), max_circuits)
            ]
        else:
            # Run as single job
            job_circuits = [circuits]

        return job_circuits


def print_summary(experiment_data: ExperimentData):
    LOG.info('Analysis results (%d total):', len(experiment_data.analysis_results()))
    for res in experiment_data.analysis_results():
        LOG.info(' - %s', res.name)

    LOG.info('Figures (%d total):', len(experiment_data.figure_names))
    for name in experiment_data.figure_names:
        LOG.info(' - %s', name)


def print_details(experiment_data: 'ExperimentData'):
    LOG.info('Analysis results (%d total):', len(experiment_data.analysis_results()))
    for res in experiment_data.analysis_results():
        LOG.info('%s', res)

    LOG.info('Figures (%d total):', len(experiment_data.figure_names))
    for name in experiment_data.figure_names:
        LOG.info(' - %s', name)
        display(experiment_data.figure(name))
