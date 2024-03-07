"""Common functions for running calibration programs."""
from argparse import ArgumentError, ArgumentParser
import datetime
import os
import pickle
import shutil
from typing import Any, Optional
import yaml
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


def setup_runner(
    backend: Backend,
    calibrations: Calibrations,
    program_config: dict[str, Any]
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

    runner = ExperimentsRunner(backend, calibrations=calibrations, data_dir=data_dir,
                               runtime_session=runtime_session)

    if (qubits := program_config.get('qubits')) is not None:
        runner.program_data['qubits'] = tuple(qubits)

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


def get_program_config(program_config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    static_required = ['base_dir', 'instance', 'backend']
    static_optional = ['name', 'qubits', 'calibrations']

    if program_config is None:
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
        parser.add_argument('-r', '--refresh-readout', action='store_true', help='Resubmit readout'
                            ' error measurements.', dest='refresh_readout')
        parser.add_argument('--read-only', action='store_true', help='Run the analyses but do not save'
                            ' the calibrations to disk.', dest='read_only')
        parser.add_argument('--dry-run', action='store_true', help='Do not submit experiment jobs; run'
                            ' the experiments with dummy data.', dest='dry_run')

        options = parser.parse_args()

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

    if not program_config['name']:
        program_config['name'] = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        program_config['name'] += f'_{program_config["backend"]}'

    return program_config
