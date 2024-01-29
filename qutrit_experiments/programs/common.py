"""Common functions for running calibration programs."""
import datetime
import os
import pickle
import shutil
from typing import Any
from qiskit.providers import Backend
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService, Session
from qiskit_ibm_runtime.api.clients import RuntimeClient
from qiskit_experiments.calibration_management import Calibrations

from ..runners import ExperimentsRunner


def setup_data_dir(program_config: dict[str, Any]):
    if not program_config.get('data_dir'):
        program_config['data_dir'] = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        program_config['data_dir'] += f'_{program_config["backend_name"]}'
    try:
        os.makedirs(
            os.path.join(
                program_config['data_dir_base'],
                program_config.get('data_dir'),
                'program_data'
            )
        )
    except FileExistsError:
        pass


def setup_backend(program_config: dict[str, Any]) -> Backend:
    if (backend := _load_backend(program_config)) is None:
        runtime_service = QiskitRuntimeService(channel='ibm_quantum',
                                               instance=program_config['provider_instance'])
        backend = runtime_service.get_backend(program_config['backend_name'],
                                              instance=program_config['provider_instance'])

        _save_backend(backend, program_config)

    return backend


def _load_backend(program_config: dict[str, Any]):
    file_name = f'backend_configurations/{program_config["backend_name"]}'
    file_name += f'-{program_config["provider_instance"].replace("/", "_")}.pkl'
    if not os.path.exists(file_name):
        return None

    with open(file_name, 'rb') as source:
        client_params, backend_config, defaults, properties, target = pickle.load(source)

    backend = IBMBackend(instance=program_config['provider_instance'],
                            configuration=backend_config,
                            api_client=RuntimeClient(client_params),
                            service=None)
    backend._defaults = defaults
    backend._properties = properties
    backend._target = target
    return backend


def _save_backend(backend: Backend, program_config: dict[str, Any]):
    backend.defaults()
    backend.properties()
    backend.target

    try:
        os.makedirs('backend_configurations')
    except FileExistsError:
        pass

    file_name = f'backend_configurations/{program_config["backend_name"]}'
    file_name += f'-{program_config["provider_instance"].replace("/", "_")}.pkl'
    with open(file_name, 'wb') as out:
        data = (
            runtime_service._client_params,
            runtime_service._backend_configs[program_config['backend_name']],
            backend.defaults(),
            backend.properties(),
            backend.target
        )
        pickle.dump(data, out)


def setup_runner(
    backend: Backend,
    calibrations: Calibrations,
    program_config: dict[str, Any]
) -> ExperimentsRunner:
    runner = ExperimentsRunner(backend, calibrations=calibrations,
                               data_dir=os.path.join(program_config['data_dir_base'],
                                                     program_config['data_dir']))

    runner.program_data['qubits'] = tuple(program_config['qubits'])

    # Submit or retrieve circuit jobs in a single process
    if program_config['session_id']:
        runner.runtime_session = Session.from_id(program_config['session_id'],
                                                 backend=program_config['backend_name'])

    return runner


def load_calibrations(
    runner: ExperimentsRunner,
    program_config: dict[str, Any]
) -> set[str]:
    calibrated = set()
    if (calib_path := program_config['calibrations_path']) is not None:
        if calib_path != '':
            shutil.copy(os.path.join(program_config['data_dir_base'], calib_path,
                                     'parameter_values.csv'),
                        runner.data_dir)
        runner.load_calibrations()
        runner.load_program_data()

        for datum in runner.calibrations.parameters_table(most_recent_only=False)['data']:
            if datum['valid'] and datum['group'] != 'default':
                calibrated.add(datum['group'])

    return calibrated