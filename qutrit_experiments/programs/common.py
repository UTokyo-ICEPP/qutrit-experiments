"""Common functions for running calibration programs."""
from collections.abc import Sequence
import os
import logging
import shutil
from typing import Any, Optional, Union

from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError
from qiskit_experiments.framework import BaseAnalysis, CompositeAnalysis
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations

from ..runners import ExperimentsRunner
from ..experiment_config import ExperimentConfigBase

logger = logging.getLogger(__name__)


def setup_data_dir(program_config: dict[str, Any]) -> str:
    data_dir = os.path.join(program_config['base_dir'], program_config.get('name'))
    try:
        os.makedirs(os.path.join(data_dir, 'program_data'))
    except FileExistsError:
        pass
    return data_dir


def setup_backend(
    program_config: dict[str, Any]
) -> Backend:
    while True:
        try:
            service = QiskitRuntimeService(channel='ibm_quantum',
                                           instance=program_config['instance'])
            return service.backend(program_config['backend'], instance=program_config['instance'])
        except IBMNotAuthorizedError:
            continue


def setup_runner(
    backend: Backend,
    program_config: dict[str, Any],
    calibrations: Optional[Calibrations] = None,
    qubits: Optional[Sequence[int]] = None,
    runner_cls: Optional[type[ExperimentsRunner]] = None
) -> ExperimentsRunner:
    data_dir = os.path.join(program_config['base_dir'], program_config['name'])
    if program_config.get('work_dir'):
        source = data_dir
        data_dir = os.path.join(program_config['base_dir'], program_config['work_dir'])
        try:
            shutil.rmtree(data_dir)
        except FileNotFoundError:
            pass
        shutil.copytree(source, data_dir)

    # Submit or retrieve circuit jobs in a single process
    if program_config['session_id']:
        # Session.from_id is not for ibm_quantum channel
        runtime_session = Session(service=backend._service, backend=backend)
        runtime_session._session_id = program_config['session_id']
    else:
        runtime_session = None

    runner_cls = runner_cls or ExperimentsRunner

    qubits = qubits or program_config.get('qubits')
    runner = runner_cls(backend, qubits=qubits, calibrations=calibrations,
                        data_dir=data_dir, runtime_session=runtime_session)

    return runner


def load_calibrations(
    runner: ExperimentsRunner,
    program_config: dict[str, Any]
) -> set[str]:
    calibrated = set()
    if (calib_path := program_config['calibrations']) is not None:
        if calib_path.endswith('.csv'):
            data_dir = os.path.dirname(calib_path)
            file_name = os.path.basename(calib_path)
        else:
            data_dir = calib_path
            file_name = 'parameter_values.csv'

        if data_dir == '':
            data_dir = None
        elif not os.path.isabs(data_dir):
            data_dir = os.path.join(program_config['base_dir'], data_dir)

        runner.load_calibrations(file_name=file_name, data_dir=data_dir)
        runner.load_program_data()

        for datum in runner.calibrations.parameters_table(most_recent_only=False)['data']:
            if datum['valid'] and datum['group'] != 'default':
                calibrated.add(datum['group'])

    return calibrated


def run_experiment(
    runner: ExperimentsRunner,
    config: Union[str, ExperimentConfigBase],
    block_for_results: bool = True,
    analyze: bool = True,
    calibrate: bool = True,
    print_level: Optional[int] = None,
    force_resubmit: bool = False,
    plot_depth: int = 0,
    parallelize: bool = True
):
    runner_qubits = set(runner.qubits)

    exp_data = None
    if runner.saved_data_exists(config):
        exp_data = runner.load_data(config)
        if (data_qubits := set(exp_data.metadata['physical_qubits'])) - runner_qubits:
            logger.warning('Saved experiment data for %s has out-of-configuration qubits.',
                           config if isinstance(config) else config.exp_type)
        runner.qubits = data_qubits

    exp = runner.make_experiment(config)
    set_analysis_option(exp.analysis, 'plot', False, threshold=plot_depth + 1)
    if not parallelize:
        set_analysis_option(exp.analysis, 'parallelize', 0)

    exp_data = runner.run_experiment(config, experiment=exp, block_for_results=block_for_results,
                                     analyze=analyze, calibrate=calibrate, print_level=print_level,
                                     exp_data=exp_data, force_resubmit=force_resubmit)

    if isinstance(exp, BaseCalibrationExperiment):
        # Exclude qubits that failed calibration
        cal_data = runner.calibrations.parameters_table(group=exp_data.experiment_type,
                                                        most_recent_only=False)['data']
        runner.qubits = runner_qubits & set(row['qubits'][0] for row in cal_data)
    else:
        runner.qubits = runner_qubits

    return exp_data


def set_analysis_option(
    analysis: BaseAnalysis,
    option: str,
    value: Any,
    threshold: int = 0,
    _depth: int = 0
):
    if _depth >= threshold and hasattr(analysis.options, option):
        analysis.set_options(**{option: value})

    if isinstance(analysis, CompositeAnalysis):
        for subanalysis in analysis.component_analysis():
            set_analysis_option(subanalysis, option, value, threshold, _depth + 1)