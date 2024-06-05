"""Initialization functions for scripts."""
from collections.abc import Sequence
import os
import logging
import shutil
from typing import Any, Optional

from qiskit.providers import Backend
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError
from qiskit_experiments.calibration_management import Calibrations

from ..runners import ExperimentsRunner

logger = logging.getLogger(__name__)


def setup_data_dir(program_config: dict[str, Any]) -> str:
    """Create the data directory and return the path."""
    data_dir = os.path.join(program_config['base_dir'], program_config.get('name'))
    try:
        os.makedirs(os.path.join(data_dir, 'program_data'))
    except FileExistsError:
        pass
    return data_dir


def setup_backend(
    program_config: dict[str, Any]
) -> Backend:
    """Obtain the backend object from QiskitRuntimeService."""
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
    """Instantiate a new ExperimentsRunner."""
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
    """Load the calibrations from a CSV or JSON file."""
    calibrated = set()
    if (calib_path := program_config['calibrations']) is not None:
        if calib_path == '':
            data_dir = None
            file_name = 'calibrations.json'
            is_csv = False
        elif os.path.isdir(calib_path):
            data_dir = calib_path
            if os.path.exists(os.path.join(data_dir, 'calibrations.json')):
                file_name = 'calibrations.json'
                is_csv = False
            elif os.path.exists(os.path.join(data_dir, 'parameter_values.csv')):
                file_name = 'parameter_values.csv'
                is_csv = True
        else:
            data_dir = os.path.dirname(calib_path)
            file_name = os.path.basename(calib_path)
            is_csv = calib_path.endswith('.csv')

        if data_dir and not os.path.isabs(data_dir):
            data_dir = os.path.join(program_config['base_dir'], data_dir)

        runner.load_calibrations(file_name=file_name, data_dir=data_dir, is_csv=is_csv)

        for datum in runner.calibrations.parameters_table(most_recent_only=False)['data']:
            if datum['valid'] and datum['group'] != 'default':
                calibrated.add(datum['group'])

    return calibrated
