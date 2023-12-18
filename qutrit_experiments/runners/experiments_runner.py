"""Standard experiments driver."""

import json
import logging
import os
import pickle
import uuid
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
from numbers import Number
from typing import TYPE_CHECKING, Optional, Union
import matplotlib
import numpy as np

from qiskit import pulse
from qiskit.circuit import Gate, Barrier
from qiskit.providers import JobStatus
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit.scheduler.config import ScheduleConfig
from qiskit.scheduler.lowering import lower_gates
from qiskit.transpiler import InstructionDurations, InstructionProperties, PassManager, Target
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, ParameterValue
from qiskit_experiments.framework import AnalysisStatus, CompositeAnalysis as CompositeAnalysisOrig
from qiskit_experiments.framework.composite.composite_experiment import CompositeExperiment
from qiskit_experiments.exceptions import AnalysisError
from qiskit_ibm_runtime import Session

from ..constants import DEFAULT_SHOTS
from ..experiment_config import ExperimentConfig, experiments
from ..framework.postprocessed_experiment_data import PostprocessedExperimentData
from ..framework.set_child_data_structure import SetChildDataStructure
from ..framework_overrides.composite_analysis import CompositeAnalysis
from ..transpilation.add_calibrations import AddQutritCalibrations
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule

if TYPE_CHECKING:
    from qiskit.providers import Backend
    from qiskit_experiments.calibration_management import Calibrations
    from qiskit_experiments.framework import BaseAnalysis, BaseExperiment, ExperimentData

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
        backend: 'Backend',
        calibrations: Optional['Calibrations'] = None,
        data_dir: Optional[str] = None,
        read_only: bool = False,
        runtime_session: Optional[Session] = None
    ):
        self._backend = backend

        # Create our own transpilation & scheduling target
        target_keys = ['description', 'num_qubits', 'dt', 'granularity', 'min_length',
                       'pulse_alignment', 'acquire_alignment', 'qubit_properties']
        self._target = Target(**{key: getattr(backend.target, key) for key in target_keys})
        for name in backend.target.operation_names:
            instruction = backend.target.operation_from_name(name)
            if isinstance(instruction, type):
                self._target.add_instruction(instruction, name=name)
            else:
                self._target.add_instruction(instruction, properties=backend.target[name],
                                             name=name)

        self._calibrations = calibrations
        if calibrations is not None:
            update_add_schedule(self._calibrations)

        self._data_dir = data_dir
        if data_dir and not os.path.exists(self._data_dir):
            os.makedirs(self._data_dir)

        self._read_only = read_only

        if runtime_session is not None:
            self._runtime_session = runtime_session
            self._set_session_start_time()
        elif self._backend.service is not None:
            self._runtime_session = Session(service=self._backend.service, backend=self._backend)
            self._session_start_time = None

        self.scheduling_triggers = []

        self.data_taking_only = False
        self.code_test = False

        self.program_data = {}

    @property
    def backend(self):
        return self._backend

    @property
    def runtime_session(self):
        if ((session := self._runtime_session) is not None
            and (max_time := session._max_time) is not None
            and (st := self._session_start_time) is not None
            and datetime.utcnow() - st > timedelta(seconds=max_time)):
            session.close()
            self._runtime_session = Session(service=self._backend.service,
                                            backend=self._backend,
                                            max_time=max_time)
            self._session_start_time = None

        return self._runtime_session

    @property
    def calibrations(self):
        return self._calibrations

    @property
    def target(self):
        return self._target

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def read_only(self):
        return self._read_only

    def load_calibrations(self):
        file_name = os.path.join(self._data_dir, 'parameter_values.csv')
        self._calibrations.load_parameter_values(file_name=file_name)

    def make_experiment(
        self,
        config: Union[str, ExperimentConfig]
    ) -> 'BaseExperiment':
        if isinstance(config, str):
            config = experiments[config](self)

        args = dict(config.args)
        args['backend'] = self._backend
        if config.physical_qubits is not None:
            args['physical_qubits'] = config.physical_qubits
        if issubclass(config.cls, BaseCalibrationExperiment):
            args['calibrations'] = self._calibrations
        if config.subexperiments:
            args['experiments'] = [self.make_experiment(sub) for sub in config.subexperiments]

        experiment = config.cls(**args)
        experiment._type = config.exp_type
        if not config.analysis:
            experiment.analysis = None

        # CompositeExperiment creates a CompositeAnalysisOrig by default; overwrite to the
        # serial version
        if type(experiment.analysis) is CompositeAnalysisOrig:
            analyses = experiment.analysis.component_analysis()
            flatten_results = experiment.analysis._flatten_results
            experiment.analysis = CompositeAnalysis(analyses, flatten_results=flatten_results)
        if isinstance(experiment, BaseCalibrationExperiment):
            experiment.auto_update = False

        return experiment

    def run_experiment(
        self,
        config: Union[str, ExperimentConfig],
        experiment: Optional['BaseExperiment'] = None,
        postprocess: bool = True,
        analyze: bool = True,
        calibrate: Union[bool, Callable[['ExperimentData'], bool]] = True,
        print_level: int = 2
    ) -> 'ExperimentData':
        if isinstance(config, str):
            config = experiments[config](self)
        if experiment is None:
            experiment = self.make_experiment(config)
        exp_type = experiment.experiment_type

        logger.info('run_experiment(%s)', exp_type)

        experiment.set_run_options(**config.run_options)

        ## Try retrieving exp_data from the pickle
        exp_data = self.find_saved_data(exp_type)
        if exp_data is None:
            # Construct the data from running or retrieving jobs -> publish the raw data
            exp_data = self._new_experiment_data(experiment)
            if self.code_test:
                logger.info('Generating dummy data for %s.', exp_type)
                self._run_code_test(experiment, exp_data)
            else:
                jobs = None
                if self._data_dir:
                    job_ids_path = os.path.join(self._data_dir, f'{exp_type}_jobs.dat')
                    if os.path.exists(job_ids_path):
                        with open(job_ids_path, encoding='utf-8') as source:
                            num_jobs = len(source.readline().strip().split())
                            instance = source.readline().strip()
                            job_unique_tag = source.readline().strip()

                        jobs = self._retrieve_jobs(job_unique_tag, num_jobs, instance)

                if not jobs:
                    jobs = self._submit_jobs(experiment)

                exp_data.add_jobs(jobs)

            for name, func in config.postprocessors:
                exp_data.add_postprocessor(name, func)
            exp_data.add_postprocessor('save_raw_data', self.save_data)

        if postprocess:
            exp_data.apply_postprocessors()
            logger.debug('Waiting for postprocessors to complete.')
            exp_data.block_for_results()

        # Status check was originally a postprocessor, but I later realized that BaseExperiment
        # cancels all analysis callbacks (which postprocessors are) when job errors are detected
        self._check_status(exp_data)

        if self._runtime_session is not None and self._session_start_time is None:
            self._set_session_start_time()

        if not analyze or experiment.analysis is None:
            if isinstance(experiment, SetChildDataStructure):
                # Usually analysis fills the child data so we do it manually instead
                SetChildDataStructure._set_child_data_structure(exp_data, fill_child_data=True)

            logger.info('No analysis will be performed for %s.', exp_type)
            return exp_data

        if type(experiment.analysis) is CompositeAnalysis:
            subanalyses = experiment.analysis.component_analysis()
            if isinstance((ana_opt := config.analysis_options), dict):
                ana_opt = [ana_opt for _ in range(len(subanalyses))]
            for subana, opt in zip(subanalyses, ana_opt):
                if opt is not None:
                    subana.set_options(**opt)
        else:
            experiment.analysis.set_options(**config.analysis_options)

        self.run_analysis(exp_data, experiment.analysis)

        if callable(calibrate):
            calibrate = calibrate(exp_data)
        if calibrate:
            self.update_calibrations(exp_data, experiment)

        if print_level == 1:
            print_summary(exp_data)
        elif print_level == 2:
            print_details(exp_data)

        return exp_data

    def run_analysis(
        self,
        experiment_data: 'ExperimentData',
        analysis: Optional['BaseAnalysis'] = None
    ):
        logger.info('Running the analysis for %s', experiment_data.experiment_type)

        if analysis is None:
            analysis = experiment_data.experiment.analysis

        analysis.run(experiment_data).block_for_results()

        if experiment_data.analysis_status() != AnalysisStatus.DONE:
            raise AnalysisError(f'Analysis status = {experiment_data.analysis_status().value}')

        self.save_data(experiment_data)

    def get_transpiled_circuits(self, experiment: 'BaseExperiment'):
        """Return a list of transpiled circuits accounting for qutrit-specific instructions."""
        if self.calibrations is None:
            # Nothing to do
            return experiment._transpiled_circuits()

        instruction_durations = InstructionDurations(backend.instruction_durations, dt=backend.dt)
        for inst_name in ['x12', 'sx12']:
            durations = [(inst_name, qubit,
                          self.calibrations.get_schedule(inst_name, qubit).duration)
                         for qubit in experiment.physical_qubits]
            instruction_durations.update(durations)
        instruction_durations.update([('rz12', qubit, 0) for qubit in experiment.physical_qubits])

        if not isinstance(experiment, BaseCalibrationExperiment):
            experiment.set_transpile_options(
                # By setting the basis_gates, PassManagerConfig.from_backend() will not take the
                # target from the backend, making target absent everywhere in the preset pass
                # manager. When the target is None, HighLevelSynthesis (responsible for translating
                # all gates to basis gates) will reference the passed basis_gates list and leaves
                # all gates appearing in the list untouched.
                basis_gates=backend.basis_gates + ['x12', 'sx12', 'rz12'],
                # Scheduling method has to be specified in case there are delay instructions that
                # violate the alignment constraints, in which case a ConstrainedRescheduling is
                # triggered, which fails without precalculated node_start_times.
                scheduling_method='alap',
                # And to run the scheduler, durations of all gates must be known.
                instruction_durations=instruction_durations
            )

        circuits = experiment._transpiled_circuits()

        # Recompute the gate durations, calculate the phase shifts for all qutrit gates, and insert
        # AC Stark shift corrections to qubit gates
        pm = PassManager()
        pm.append(ALAPScheduleAnalysis(instruction_durations))
        add_cal = AddQutritCalibrations(self.backend.target)
        add_cal.calibrations = self.calibrations # See the comment in the class for why we do this
        pm.append(add_cal)
        circuits = pm.run(circuits)

        return circuits

    def save_data(
        self,
        experiment_data: 'ExperimentData'
    ):
        if not self._data_dir:
            return

        pickle_name = os.path.join(self._data_dir, f'{experiment_data.experiment_type}.pkl')
        with open(pickle_name, 'wb') as out:
            pickle.dump(experiment_data, out)

    def update_calibrations(
        self,
        experiment_data: 'ExperimentData',
        experiment: Optional['BaseExperiment'] = None
    ):
        def _get_update_list(exp_data, exp):
            if isinstance(exp, BaseCalibrationExperiment):
                exp.update_calibrations(exp_data)

                param_name = exp._param_name
                sched_name = exp._sched_name
                if isinstance(param_name, str):
                    return [(param_name, sched_name, exp.physical_qubits)]
                if isinstance(sched_name, str):
                    return [(pname, sched_name, exp.physical_qubits) for pname in param_name]

                return [(pname, sname, exp.physical_qubits)
                        for pname, sname in zip(param_name, sched_name)]

            if isinstance(exp, CompositeExperiment):
                update_list = []

                for subexp, child_data in zip(exp.component_experiment(), exp_data.child_data()):
                    update_list.extend(_get_update_list(child_data, subexp))

                return update_list

            return []

        if experiment is None:
            experiment = experiment_data.experiment

        update_list = _get_update_list(experiment_data, experiment)

        self.save_calibrations(experiment.experiment_type, update_list)

        target_update_list = []
        for _, sname, qubits in update_list:
            try:
                self._target[sname][qubits]
            except KeyError:
                continue

            target_update_list.append((sname, qubits))

        self.update_target(target_update_list)

    def save_calibrations(
        self,
        exp_type: str,
        update_list: list[tuple[str, str, tuple[int, ...]]]
    ):
        for pname, sname, qubits in update_list:
            logger.info('Tagging calibration parameter %s:%s:%s from experiment %s',
                        pname, sname, qubits, exp_type)

            self.pass_parameter_value(pname, qubits, from_schedule=sname, from_group='default',
                                      to_group=exp_type)

        if update_list and self._data_dir and not self._read_only:
            self._calibrations.save(folder=self._data_dir, overwrite=True)

    def update_target(
        self,
        update_list: Optional[list[tuple[str, tuple[int, ...]]]] = None
    ):
        if update_list is None:
            update_list = []
            for item in self._calibrations.schedules():
                sname = item['schedule'].name
                qubits = item['qubits']
                if self._target.has_calibration(sname, qubits):
                    update_list.append((sname, qubits))

        for sname, qubits in update_list:
            current = self._target[sname][qubits]
            # Not handling the case of parametric schedules for now
            schedule = self.calibrations.get_schedule(sname, qubits)
            properties = InstructionProperties(duration=current.duration, error=current.error,
                                               calibration=schedule)
            self._target.update_instruction_properties(sname, qubits, properties)

    def saved_data_exists(self, exp_type: str) -> bool:
        if not self._data_dir:
            return False

        pickle_path = os.path.join(self._data_dir, f'{exp_type}.pkl')
        return os.path.exists(pickle_path)

    def find_saved_data(self, exp_type: str) -> Union['ExperimentData', None]:
        if not self.saved_data_exists(exp_type):
            return None

        pickle_path = os.path.join(self._data_dir, f'{exp_type}.pkl')

        with open(pickle_path, 'rb') as source:
            exp_data = deep_copy_no_results(pickle.load(source))

        logger.info('Unpickled experiment data for %s.', exp_type)

        return exp_data

    def _new_experiment_data(
        self,
        experiment: 'BaseExperiment',
    ) -> 'ExperimentData':
        # Finalize the experiment before executions
        experiment._finalize()

        # Initialize the result container
        # This line comes after _transpiled_circuits in the original BaseExperiment.run() (see
        # below) but in terms of good object design there shouldn't be any dependencies between the
        # two methods
        experiment_data = experiment._initialize_experiment_data()
        if not isinstance(experiment_data, PostprocessedExperimentData):
            experiment_data = PostprocessedExperimentData(experiment=experiment)

        return experiment_data

    def _retrieve_jobs(
        self,
        job_unique_tag,
        num_jobs,
        instance
    ):
        # provider.jobs() is a lot faster than calling retrieve_job multiple times
        logger.info('Reconstructing experiment data using the unique id %s',
                    job_unique_tag)

        if (provider := self._backend.provider) is None:
            provider = self._backend.service

        return provider.jobs(limit=num_jobs, backend_name=self._backend.name,
                             job_tags=[job_unique_tag], instance=instance)

    def _submit_jobs(
        self,
        experiment: 'BaseExperiment',
    ) -> 'ExperimentData':
        logger.info('Submitting experiment circuit jobs for %s.', experiment.experiment_type)

        # Generate and transpile circuits
        transpiled_circuits = self.get_transpiled_circuits(experiment)

        # Run options
        run_opts = {**experiment.run_options.__dict__}

        job_tags = None
        if type(self._backend).__name__ == 'IBMBackend':
            instance = self._backend._instance
            job_unique_tag = uuid.uuid4().hex
            # Add a tag to the job to make later identification easier
            job_tags = [experiment.experiment_type, job_unique_tag]

        # Run jobs (reimplementing BaseExperiment._run_jobs because we need to a handle
        # on job splitting)
        circuit_lists = self._split_circuits(experiment, transpiled_circuits)

        jobs = []

        # Check all jobs got an id. If not, try resubmitting
        for ilist, circs in enumerate(circuit_lists):
            for _ in range(5):
                if (session := self.runtime_session) is not None:
                    options = {'instance': self._backend._instance}
                    if job_tags:
                        options['job_tags'] = job_tags

                    inputs = {'circuits': circs, 'skip_transpilation': True, **run_opts}
                    job = session.run('circuit-runner', inputs, options=options)
                else:
                    job = self._backend.run(circs, job_tags=job_tags, **run_opts)

                if job.job_id():
                    jobs.append(job)
                    break
            else:
                start = sum(len(lst) for lst in circuit_lists[:ilist])
                end = start + len(circs) - 1
                raise RuntimeError(f'Failed to submit circuits {start}-{end}')

        logger.info('Job IDs: %s', [job.job_id() for job in jobs])

        if self._data_dir:
            job_ids_path = os.path.join(self._data_dir, f'{experiment.experiment_type}_jobs.dat')
            with open(job_ids_path, 'w', encoding='utf-8') as out:
                out.write(' '.join(job.job_id() for job in jobs) + '\n')
                out.write(f'{instance}\n{job_unique_tag}\n')

        return jobs

    def _check_status(self, experiment_data: 'ExperimentData'):
        logger.info('Checking the job status for %s', experiment_data.experiment_type)

        ## Check the job status
        job_status = experiment_data.job_status()
        if job_status != JobStatus.DONE:
            if self._data_dir:
                job_ids_path = os.path.join(self._data_dir,
                                            f'{experiment_data.experiment_type}_jobs.dat')
                failed_jobs_path = os.path.join(self._data_dir,
                                                f'{experiment_data.experiment_type}_failedjobs.dat')
                try:
                    with open(failed_jobs_path, 'a', encoding='utf-8') as out:
                        with open(job_ids_path, 'r', encoding='utf-8') as source:
                            out.write(source.read())

                    os.unlink(job_ids_path)
                except Exception:
                    pass

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

    def _set_session_start_time(self):
        first_job_id = self._runtime_session.session_id
        if first_job_id is None:
            return

        api_client = self._runtime_session.service._api_client
        job_metadata = api_client.job_metadata(first_job_id)
        if isinstance(job_metadata, dict):
            # qiskit-ibm-provider >= 0.6?
            timestamps = job_metadata['timestamps']
        else:
            timestamps = json.loads(job_metadata)['timestamps']

        try:
            self._session_start_time = datetime.strptime(timestamps['running'],
                                                         '%Y-%m-%dT%H:%M:%S.%fZ')
        except (KeyError, TypeError):
            self._session_start_time = None

    def _run_code_test(self, experiment, experiment_data):
        """Test circuit generation and then fill the container with dummy data."""
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

        parameter_values = self._calibrations._params[(from_param, from_qubits, from_schedule)]

        parameter_values = list(pv for pv in parameter_values if pv.group == from_group)

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

        to_value = ParameterValue(value, from_value.date_time, group=to_group)
        self._calibrations.add_parameter_value(to_value, to_param, qubits=to_qubits,
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


def deep_copy_no_results(experiment_data: 'ExperimentData'):
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


def print_summary(experiment_data: 'ExperimentData'):
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
