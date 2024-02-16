"""Common functions for running calibration programs."""
from argparse import ArgumentError, ArgumentParser
import datetime
import os
import pickle
import shutil
import yaml
from typing import Any
from qiskit.providers import Backend
from qiskit_ibm_runtime import IBMBackend, QiskitRuntimeService, Session
from qiskit_ibm_runtime.api.clients import RuntimeClient
from qiskit_experiments.calibration_management import Calibrations

from ..runners import ExperimentsRunner


def setup_data_dir(program_config: dict[str, Any]) -> str:
    data_dir = os.path.join(program_config['base_dir'], program_config.get('name'))
    try:
        os.makedirs(os.path.join(data_dir, 'program_data'))
    except FileExistsError:
        pass
    return data_dir


def setup_backend(program_config: dict[str, Any]) -> Backend:
    if (backend := _load_backend(program_config)) is None:
        runtime_service = QiskitRuntimeService(channel='ibm_quantum',
                                               instance=program_config['instance'])
        backend = runtime_service.get_backend(program_config['backend'],
                                              instance=program_config['instance'])

        _save_backend(backend, program_config)

    return backend


def _load_backend(program_config: dict[str, Any]):
    file_name = os.path.join(
        program_config['base_dir'],
        'backend_configurations',
        f'{program_config["backend"]}-{program_config["instance"].replace("/", "_")}.pkl'
    )
    if not os.path.exists(file_name):
        return None

    with open(file_name, 'rb') as source:
        client_params, backend_config, defaults, properties, target = pickle.load(source)

    backend = IBMBackend(instance=program_config['instance'],
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

    file_name = os.path.join(
        program_config['base_dir'],
        'backend_configurations',
        f'{program_config["backend"]}-{program_config["instance"].replace("/", "_")}.pkl'
    )

    try:
        os.makedirs(os.path.dirname(file_name))
    except FileExistsError:
        pass
    
    with open(file_name, 'wb') as out:
        data = (
            backend.service._client_params,
            backend.service._backend_configs[program_config['backend']],
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
                               data_dir=os.path.join(program_config['base_dir'],
                                                     program_config['name']))

    if (qubits := program_config.get('qubits')) is not None:
        runner.program_data['qubits'] = tuple(qubits)

    # Submit or retrieve circuit jobs in a single process
    if program_config['session_id']:
        runner.runtime_session = Session.from_id(program_config['session_id'],
                                                 backend=program_config['backend'])

    return runner


def load_calibrations(
    runner: ExperimentsRunner,
    program_config: dict[str, Any]
) -> set[str]:
    calibrated = set()
    if (calib_path := program_config['calibrations']) is not None:
        if calib_path != '':
            shutil.copy(os.path.join(program_config['base_dir'], calib_path,
                                     'parameter_values.csv'),
                        runner.data_dir)
        runner.load_calibrations()
        runner.load_program_data()

        for datum in runner.calibrations.parameters_table(most_recent_only=False)['data']:
            if datum['valid'] and datum['group'] != 'default':
                calibrated.add(datum['group'])

    return calibrated


def get_program_config():
    parser = ArgumentParser(
        prog='single_qutrit_gates',
        description='Calibrate and characterize the X12 and SX12 gates.'
    )
    parser.add_argument('-f', '--config-file', help='Configuration yaml file. The following'
                        ' command-line options are ignored when a configuration file is present:'
                        ' --name, --base-dir, --instance, --backend, --qubits, --calibrations.',
                        metavar='PATH', dest='config_file')
    parser.add_argument('-w', '--work-dir', help='Scratch working directory. The contents of the'
                        ' directory are removed before the program execution.', metavar='PATH',
                        dest='work_dir')
    parser.add_argument('-n', '--name', help='Identifier for this program instance.',
                        metavar='NAME', dest='name')
    parser.add_argument('-d', '--base-dir', help='Base data directory path.', metavar='BASE_DIR',
                        dest='base_dir')
    parser.add_argument('-i', '--instance', help='IBM Quantum service instance.',
                        metavar='HUB/PROJECT/GROUP', dest='instance')
    parser.add_argument('-b', '--backend', help='IBM Quantum backend name.', metavar='NAME',
                        dest='backend')
    parser.add_argument('-q', '--qubits', nargs='+', type=int, help='Qubits to use.', dest='qubits')
    parser.add_argument('-c', '--calibrations', nargs='?', const='', help='Load calibrations. If a'
                        ' path is given, data is loaded from BASE_DIR/PATH/parameter_values.csv.')
    parser.add_argument('-s', '--session-id', help='Qiskit Runtime Session ID.', metavar='ID',
                        dest='session_id')
    parser.add_argument('--read-only', action='store_true', help='Run the analyses but do not save'
                        ' the calibrations to disk.', dest='read_only')
    parser.add_argument('--dry-run', action='store_true', help='Do not submit experiment jobs; run'
                        ' the experiments with dummy data.', dest='dry_run')
    
    options = parser.parse_args()
    static_required = ['base_dir', 'instance', 'backend']
    static_optional = ['name', 'qubits', 'calibrations']
    if options.config_file is not None:
        with open(options.config_file, 'r') as source:
            program_config = yaml.safe_load(source)
        for key, value in vars(options).items():
            if key not in static_required + static_optional:
                program_config[key] = value
    else:
        program_config = vars(options)
    
    if any(value is None for key, value in program_config.items() if key in static_required):
        raise RuntimeError('One or more of the following options are missing:'
                           f' {["config_file"] + static_required}')
    
    if program_config['name'] is None:
        program_config['name'] = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        program_config['name'] += f'_{program_config["backend"]}'

    return program_config
