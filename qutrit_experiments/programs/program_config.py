from argparse import ArgumentParser
import datetime
from typing import Any, Optional
import yaml

def get_program_config(
    program_config: Optional[dict[str, Any]] = None,
    additional_args: Optional[list[tuple[tuple, dict]]] = None
) -> dict[str, Any]:
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
        parser.add_argument('-q', '--qubits', nargs='+', default=[], help='Qubits to use.',
                            dest='qubits')
        parser.add_argument('-c', '--calibrations', nargs='?', const='', help='Load calibrations.'
                            ' If a path is given, data is loaded from BASE_DIR/PATH/parameter_values.csv.'
                            ' If the path ends with .csv, it is considered to be the path to the'
                            ' CSV file (absolute if it starts with /, else relative to BASE_DIR'
                            ' if it contains a /, else under BASE_DIR/PATH).')
        parser.add_argument('-s', '--session-id', help='Qiskit Runtime Session ID.', metavar='ID',
                            dest='session_id')
        parser.add_argument('-r', '--refresh-readout', action='store_true', help='Resubmit readout'
                            ' error measurements.', dest='refresh_readout')
        parser.add_argument('--read-only', action='store_true', help='Run the analyses but do not save'
                            ' the calibrations to disk.', dest='read_only')
        parser.add_argument('--dry-run', action='store_true', help='Do not submit experiment jobs; run'
                            ' the experiments with dummy data.', dest='dry_run')
        
        for args, kwargs in (additional_args or []):
            parser.add_argument(*args, **kwargs)

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

    qubits = []
    for qarg in program_config['qubits']:
        if isinstance(qarg, int):
            qubits.append(qarg)
        elif (parts := qarg.partition('-'))[1] == '-':
            qubits.extend(range(int(parts[0]), int(parts[2]) + 1))
        else:
            qubits.append(int(qarg))
    program_config['qubits'] = qubits

    return program_config
