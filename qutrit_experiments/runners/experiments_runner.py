"""Standard experiments driver."""

import logging
import os
import pickle
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from numbers import Number
from typing import Optional, Union
import matplotlib
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend, JobStatus
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.framework import (AnalysisStatus, BaseExperiment,
                                          CompositeAnalysis as CompositeAnalysisOrig,
                                          ExperimentData)
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_ibm_runtime import RuntimeJob, Session
from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError, IBMRuntimeError

from ..constants import DEFAULT_REP_DELAY, DEFAULT_SHOTS, RESTLESS_REP_DELAY
from ..experiment_config import (CompositeExperimentConfig, ExperimentConfigBase,
                                 experiments, postexperiments)
from ..framework.child_data import set_child_data_structure, fill_child_data
from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework_overrides.composite_analysis import CompositeAnalysis
from ..framework_overrides.parallel_experiment import ParallelExperiment
from ..transpilation.qutrit_transpiler import (BASIS_GATES, QutritTranspileOptions,
                                               make_instruction_durations, transpile_qutrit_circuits)

def display(_): # pylint: disable=missing-function-docstring
    pass
if 'backend_inline' in matplotlib.rcParams['backend']:
    try:
        from IPython.display import display
    except ImportError:
        pass

logger = logging.getLogger(__name__)


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
        calibrations: Optional[Calibrations] = None,
        data_dir: Optional[str] = None,
        read_only: bool = False,
        runtime_session: Optional[Session] = None
    ):
        self._backend = backend
        self.calibrations = calibrations

        self.qubits = qubits
        self.data_dir = data_dir

        self.read_only = read_only
        self.default_print_level = 2

        if runtime_session is not None:
            self._runtime_session = runtime_session
        elif not getattr(self._backend, 'simulator', True):
            self._runtime_session = Session(service=self._backend.service, backend=self._backend)
        else:
            self._runtime_session = None

        self._skip_missing_calibration = False

        self.data_taking_only = False
        self.code_test = False
        self.job_retry_interval = -1.

        self.qutrit_transpile_options = QutritTranspileOptions()

        self.program_data = {}

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, backend: Backend):
        self._backend = backend
        self.runtime_session = Session(service=self._backend.service, backend=self._backend)

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

    @property
    def runtime_session(self):
        if (session := self._runtime_session) is not None and session.status() == 'Closed':
            session.close()
            self._runtime_session = Session(service=self._backend.service, backend=self._backend)
        return self._runtime_session

    @runtime_session.setter
    def runtime_session(self, session: Session):
        self._runtime_session = session

    def load_calibrations(
        self,
        file_name: str = 'parameter_values.csv',
        data_dir: Optional[str] = None
    ):
        if data_dir is None:
            data_dir = self._data_dir
        self.calibrations.load_parameter_values(file_name=os.path.join(data_dir, file_name))

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

        args = {'backend': self._backend}
        if isinstance(config, CompositeExperimentConfig):
            args['experiments'] = [self._make_experiment(sub) for sub in config.subexperiments]
        else:
            args.update(config.args)
            if config.physical_qubits is not None:
                args['physical_qubits'] = config.physical_qubits
            if issubclass(config.cls, BaseCalibrationExperiment):
                args['calibrations'] = self.calibrations

        experiment = config.cls(**args)
        experiment._type = config.exp_type
        if not config.analysis:
            experiment.analysis = None

        experiment.set_experiment_options(**config.experiment_options)
        if isinstance(config, CompositeExperimentConfig):
            # CompositeExperiment creates a CompositeAnalysisOrig by default; overwrite to the
            # serial version
            if type(experiment.analysis) is CompositeAnalysisOrig: # pylint: disable=unidiomatic-typecheck
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

        logger.info('run_experiment(%s)', exp_type)

        experiment.set_run_options(**config.run_options)

        if force_resubmit:
            self.delete_data(exp_type)

        if exp_data is None:
            ## Try retrieving exp_data from the pickle
            if self.saved_data_exists(exp_type):
                exp_data = self.load_data(exp_type)
            else:
                # Construct the data from running or retrieving jobs -> publish the raw data
                # Finalize the experiment before executions
                experiment._finalize()
                # Initialize the result container
                # This line comes after _transpiled_circuits in the original BaseExperiment.run() (see
                # below) but in terms of good object design there shouldn't be any dependencies between the
                # two methods
                exp_data = experiment._initialize_experiment_data()

                if self.code_test:
                    self._run_code_test(experiment, exp_data)
                else:
                    if self.job_ids_exist(exp_type):
                        jobs = self.load_jobs(exp_type)
                    else:
                        jobs = self._submit_jobs(experiment)

                    exp_data.add_jobs(jobs)

                with exp_data._analysis_callbacks.lock:
                    if isinstance(experiment, CompositeExperiment):
                        exp_data.add_analysis_callback(set_child_data_structure)
                    exp_data.add_analysis_callback(self.save_data)

        if block_for_results:
            # Wait here once just to make the program execution easier to understand
            exp_data.block_for_results()
            # Analysis callbacks get cleared when there are job errors; need to call check_status
            # here directly to raise upon error
            self._check_job_status(exp_data)
        else:
            # Which means this is probably meaningless
            with exp_data._analysis_callbacks.lock:
                exp_data.add_analysis_callback(self._check_job_status)

        if not analyze or experiment.analysis is None:
            if isinstance(experiment, CompositeExperiment):
                # Usually analysis fills the child data so we do it manually instead
                with exp_data._analysis_callbacks.lock:
                    exp_data.add_analysis_callback(fill_child_data)

            if block_for_results:
                self._check_status(exp_data)

            logger.info('No analysis will be performed for %s.', exp_type)
            return exp_data

        if block_for_results:
            logger.info('Running the analysis for %s', exp_type)
        else:
            logger.info('Reserving the analysis for %s', exp_type)
        experiment.analysis.run(exp_data)

        if block_for_results:
            self._check_status(exp_data)

        if calibrate and self.calibrations is not None:
            def update_calibrations(exp_data):
                self.update_calibrations(exp_data, experiment=experiment,
                                         criterion=config.calibration_criterion)
            with exp_data._analysis_callbacks.lock:
                exp_data.add_analysis_callback(update_calibrations)

        if (postexp := postexperiments.get(exp_type)) is not None:
            if block_for_results:
                self._check_status(exp_data)
                logger.info('Performing the postexperiment for %s.', exp_type)
            else:
                logger.info('Reserving the postexperiment for %s.', exp_type)
            def postexperiment(exp_data):
                postexp(self, exp_data)
            with exp_data._analysis_callbacks.lock:
                exp_data.add_analysis_callback(postexperiment)

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

        instruction_durations = make_instruction_durations(self._backend, self.calibrations,
                                                           qubits=experiment.physical_qubits)

        if not transpiled_circuits:
            if not isinstance(experiment, BaseCalibrationExperiment):
                experiment.set_transpile_options(
                    # By setting the basis_gates, PassManagerConfig.from_backend() will not take the
                    # target from the backend, making target absent everywhere in the preset pass
                    # manager. When the target is None, HighLevelSynthesis (responsible for translating
                    # all gates to basis gates) will reference the passed basis_gates list and leaves
                    # all gates appearing in the list untouched.
                    basis_gates=self._backend.basis_gates + [g.gate_name for g in BASIS_GATES],
                    # Scheduling method has to be specified in case there are delay instructions that
                    # violate the alignment constraints, in which case a ConstrainedRescheduling is
                    # triggered, which fails without precalculated node_start_times.
                    scheduling_method='alap',
                    # And to run the scheduler, durations of all gates must be known.
                    instruction_durations=instruction_durations
                )

            start = time.time()
            transpiled_circuits = experiment._transpiled_circuits()
            end = time.time()
            logger.debug('Initial transpilation took %.1f seconds.', end - start)

        start = time.time()
        transpiled_circuits = transpile_qutrit_circuits(transpiled_circuits,
                                                        self._backend, self.calibrations,
                                                        qubit_transpiled=True,
                                                        instruction_durations=instruction_durations,
                                                        options=self.qutrit_transpile_options)
        end = time.time()
        logger.debug('Qutrit-specific transpilation took %.1f seconds.', end - start)
        return transpiled_circuits

    def save_data(
        self,
        experiment_data: ExperimentData
    ):
        if not self._data_dir:
            return
        # _experiment and backend attributes may contain unpicklable objects
        experiment = experiment_data.experiment
        backend = experiment_data.backend
        experiment_data._experiment = None
        experiment_data.backend = None

        def _convert_memory_to_array(data):
            for datum in data.data():
                if 'memory' in datum:
                    datum['memory'] = np.array(datum['memory'])
            for child_data in data.child_data():
                _convert_memory_to_array(child_data)

        _convert_memory_to_array(experiment_data)

        pickle_name = os.path.join(self._data_dir, f'{experiment_data.experiment_type}.pkl')
        with open(pickle_name, 'wb') as out:
            pickle.dump(deep_copy_no_results(experiment_data), out)
        experiment_data._experiment = experiment
        experiment_data.backend = backend

    def saved_data_exists(self, exp_type: str) -> bool:
        return self._file_exists(f'{exp_type}.pkl')

    def delete_data(self, exp_type: str):
        for suffix in ['_jobs.dat', '_failedjobs.dat', '.pkl']:
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

    def load_data(self, exp_type: str) -> ExperimentData:
        if not self._data_dir:
            raise RuntimeError('ExperimentsRunner does not have a data_dir set')
        start = time.time()
        with open(os.path.join(self._data_dir, f'{exp_type}.pkl'), 'rb') as source:
            exp_data = deep_copy_no_results(pickle.load(source))

        def _convert_memory_to_list(data):
            for datum in data.data():
                if 'memory' in datum:
                    datum['memory'] = datum['memory'].tolist()
            for child_data in data.child_data():
                _convert_memory_to_list(child_data)

        _convert_memory_to_list(exp_data)
        end = time.time()

        logger.info('Unpickled experiment data for %s.', exp_type)
        logger.debug('Loading data required %.1f seconds.', end - start)
        return exp_data

    def load_jobs(self, exp_type: str) -> list[RuntimeJob]:
        if not self._data_dir:
            raise RuntimeError('ExperimentsRunner does not have a data_dir set')
        with open(os.path.join(self._data_dir, f'{exp_type}_jobs.dat'), encoding='utf-8') as source:
            num_jobs = len(source.readline().strip().split())
            job_unique_tag = source.readline().strip()

        # provider.jobs() is a lot faster than calling retrieve_job multiple times
        logger.info('Retrieving runtime job with unique id %s', job_unique_tag)
        while True:
            try:
                return self._runtime_session.service.jobs(limit=num_jobs,
                                                          backend_name=self._backend.name,
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
        if exp_type is None:
            exp_type = experiment.experiment_type

        logger.info('Updating calibrations for %s.', exp_type)

        def _get_update_list(exp_data, exp):
            update_list = []
            if isinstance(exp, BaseCalibrationExperiment):
                updated = False
                try:
                    if criterion and not criterion(exp_data):
                        logger.warning('%s qubits %s failed calibration criterion', exp_type,
                                       exp.physical_qubits)
                    else:
                        exp.update_calibrations(exp_data)
                        updated = True
                except ExperimentEntryNotFound as exc:
                    if self._skip_missing_calibration:
                        logger.warning('%s (%s) %s', exp_type, exp.physical_qubits, exc.message)
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
                    update_list.append((pname, sname, qubits, updated))

            elif type(exp) in [BatchExperiment, ParallelExperiment]:
                for subexp, child_data in zip(exp.component_experiment(), exp_data.child_data()):
                    update_list.extend(_get_update_list(child_data, subexp))

            return update_list

        update_list = _get_update_list(experiment_data, experiment)
        logger.info('%d/%d parameters to update.',
                    len([x for x in update_list if x[-1]]), len(update_list))

        for pname, sname, qubits, updated in update_list:
            if not updated:
                continue

            logger.debug('Tagging calibration parameter %s:%s:%s from experiment %s',
                         pname, sname, qubits, exp_type)
            self.pass_parameter_value(pname, qubits, from_schedule=sname, from_group='default',
                                      to_group=exp_type)

        if any(x[-1] for x in update_list) and self._data_dir and not self.read_only:
            self.calibrations.save(folder=self._data_dir, overwrite=True)

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
            with open(os.path.join(pdata_path, f'{key}.pkl'), 'wb') as out:
                pickle.dump(self.program_data[key], out)

    def load_program_data(
        self,
        keys: Optional[Union[str, list[str]]] = None,
        allow_missing: bool = False
    ):
        if (not self._data_dir
            or not os.path.isdir(pdata_path := os.path.join(self._data_dir, 'program_data'))):
            raise RuntimeError('Program data directory does not exist')
        if not keys:
            keys = map(lambda s: s.replace('.pkl', ''), os.listdir(pdata_path))
        elif isinstance(keys, str):
            keys = [keys]
        for key in keys:
            try:
                with open(os.path.join(pdata_path, f'{key}.pkl'), 'rb') as source:
                    self.program_data[key] = pickle.load(source)
            except FileNotFoundError:
                if allow_missing:
                    logger.info('Program data %s not found.', key)
                else:
                    raise

    def _submit_jobs(
        self,
        experiment: BaseExperiment,
    ) -> ExperimentData:
        logger.info('Creating transpiled circuits for %s.', experiment.experiment_type)
        # Generate and transpile circuits
        start = time.time()
        transpiled_circuits = self.get_transpiled_circuits(experiment)
        end = time.time()
        logger.debug('Created %d circuits for %s in %.1f seconds.',
                     len(transpiled_circuits), experiment.experiment_type, end - start)

        # Run options
        run_opts = {**experiment.run_options.__dict__}
        if 'shots' not in run_opts:
            run_opts['shots'] = DEFAULT_SHOTS
        if 'rep_delay' not in run_opts:
            run_opts['rep_delay'] = DEFAULT_REP_DELAY

        job_unique_tag = uuid.uuid4().hex
        # Add a tag to the job to make later identification easier
        job_tags = [experiment.experiment_type, job_unique_tag]

        # Run jobs (reimplementing BaseExperiment._run_jobs because we need to a handle
        # on job splitting)
        circuit_lists = self._split_circuits(experiment, transpiled_circuits)
        logger.debug('Circuits will be split into %d jobs.', len(circuit_lists))

        jobs = []

        if not getattr(self._backend, 'simulator', True):
            logger.info('Submitting experiment circuit jobs for %s.', experiment.experiment_type)
            # Check all jobs got an id. If not, try resubmitting
            for ilist, circs in enumerate(circuit_lists):
                for _ in range(5):
                    options = {'instance': self._backend._instance, 'job_tags': job_tags}
                    inputs = {'circuits': circs, 'skip_transpilation': True, **run_opts}
                    while True:
                        try:
                            job = self.runtime_session.run('circuit-runner', inputs,
                                                           options=options)
                            break
                        except (IBMRuntimeError, IBMNotAuthorizedError) as ex:
                            if self.job_retry_interval < 0.:
                                raise
                            else:
                                logger.error('IBMRuntimeError during job submission: %s', ex.message)

                        time.sleep(self.job_retry_interval)

                    if job.job_id():
                        jobs.append(job)
                        break
                else:
                    start = sum(len(lst) for lst in circuit_lists[:ilist])
                    end = start + len(circs) - 1
                    raise RuntimeError(f'Failed to submit circuits {start}-{end}')
        else:
            jobs = [self._backend.run(circs, job_tags=job_tags, **run_opts)
                    for circs in circuit_lists]

        logger.info('Job IDs: %s', [job.job_id() for job in jobs])

        if self._data_dir:
            job_ids_path = os.path.join(self._data_dir, f'{experiment.experiment_type}_jobs.dat')
            with open(job_ids_path, 'w', encoding='utf-8') as out:
                out.write(' '.join(job.job_id() for job in jobs) + '\n')
                out.write(job_unique_tag + '\n')

        return jobs

    def _check_job_status(self, experiment_data: ExperimentData):
        logger.info('Checking the job status of %s', experiment_data.experiment_type)

        ## Check the job status
        if (job_status := experiment_data.job_status()) != JobStatus.DONE:
            if self._data_dir:
                job_ids_path = os.path.join(self._data_dir,
                                            f'{experiment_data.experiment_type}_jobs.dat')
                failed_jobs_path = os.path.join(self._data_dir,
                                                f'{experiment_data.experiment_type}_failedjobs.dat')
                with open(failed_jobs_path, 'a', encoding='utf-8') as out:
                    with open(job_ids_path, 'r', encoding='utf-8') as source:
                        out.write(source.read())

                os.unlink(job_ids_path)

            raise RuntimeError(f'Job status = {job_status.value}')

        ## Fix a bug in ExperimentData (https://github.com/Qiskit/qiskit-experiments/issues/963)
        now = datetime.now(timezone.utc).astimezone()
        for job in experiment_data._jobs.values():
            # job can be None if the added job data has an unknown job id for some reason
            if job and not hasattr(job, 'time_per_step'):
                if job.done():
                    job.time_per_step = lambda: {'COMPLETED': now}
                else:
                    job.time_per_step = lambda: {}

    def _check_status(self, experiment_data: ExperimentData):
        logger.debug('Checking the status of %s', experiment_data.experiment_type)
        experiment_data.block_for_results()
        if (status := experiment_data.analysis_status()) != AnalysisStatus.DONE:
            raise RuntimeError(f'Post-job status = {status.value}')

    def _run_code_test(self, experiment, experiment_data):
        """Test circuit generation and then fill the container with dummy data."""
        logger.info('Generating dummy data for %s.', experiment.experiment_type)
        run_opts = experiment.run_options.__dict__

        shots = run_opts.get('shots', DEFAULT_SHOTS)
        meas_level = run_opts['meas_level']
        meas_return = run_opts.get('meas_return')

        result_data_template = {
            'shots': shots,
            'meas_level': meas_level,
            'job_id': 'dummy'
        }
        if meas_return is not None:
            result_data_template['meas_return'] = meas_return

        circuits = self.get_transpiled_circuits(experiment)

        dummy_metadata = [c.metadata for c in circuits]

        try:
            dummy_payload = experiment.dummy_data(circuits)
        except AttributeError:
            dummy_payload = []

            for circuit in circuits:
                if meas_level == MeasLevel.CLASSIFIED:
                    counts = Counts({('0' * circuits.num_clbits): shots})
                    dummy_payload.append(counts)
                else:
                    if meas_return == 'single':
                        dummy_payload.append(np.zeros((shots, circuit.num_clbits, 2)))
                    else:
                        dummy_payload.append(np.zeros((circuit.num_clbits, 2)))

        with experiment_data._result_data.lock:
            for payload, metadata in zip(dummy_payload, dummy_metadata):
                result_data = dict(result_data_template)
                if meas_level == MeasLevel.CLASSIFIED:
                    result_data['counts'] = payload
                else:
                    result_data['memory'] = payload.tolist()
                result_data['metadata'] = metadata

                experiment_data._result_data.append(result_data)

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

        logger.info('Adding parameter value %s for %s qubits=%s schedule=%s',
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
                circuits[i : i + max_circuits] for i in range(0, len(circuits), max_circuits)
            ]
        else:
            # Run as single job
            job_circuits = [circuits]

        return job_circuits


def deep_copy_no_results(experiment_data: ExperimentData):
    """Create a shallow copy of the experiment data, including child data.

    Serialized ExperimentData is missing the attribute _monitor_executor due to a bug.
    The cleanest workaround would be to create a fresh instance through data.copy(),
    but data.copy(copy_results=True) is extremely slow and data.copy(copy_results=False)
    does not copy the child data. We therefore need a recursive function to copy the raw
    data through the full parent-children hierarchy.
    """
    new_data = experiment_data.copy(copy_results=False)
    new_child_data = [deep_copy_no_results(child) for child in experiment_data.child_data()]
    new_data._set_child_data(new_child_data)

    return new_data


def print_summary(experiment_data: ExperimentData):
    logger.info('Analysis results (%d total):', len(experiment_data.analysis_results()))
    for res in experiment_data.analysis_results():
        logger.info(' - %s', res.name)

    logger.info('Figures (%d total):', len(experiment_data.figure_names))
    for name in experiment_data.figure_names:
        logger.info(' - %s', name)


def print_details(experiment_data: 'ExperimentData'):
    logger.info('Analysis results (%d total):', len(experiment_data.analysis_results()))
    for res in experiment_data.analysis_results():
        logger.info('%s', res)

    logger.info('Figures (%d total):', len(experiment_data.figure_names))
    for name in experiment_data.figure_names:
        logger.info(' - %s', name)
        display(experiment_data.figure(name))
