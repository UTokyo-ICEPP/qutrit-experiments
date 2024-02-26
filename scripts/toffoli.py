#!/usr/bin/env python

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qutrit_experiments.calibrations import (make_single_qutrit_gate_calibrations,
                                                 make_qutrit_qubit_cx_calibrations)
    import qutrit_experiments.configurations.toffoli
    from qutrit_experiments.programs.common import (get_program_config, load_calibrations,
                                                    setup_backend, setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.programs.qutrit_qubit_cx import calibrate_qutrit_qubit_cx


    program_config = get_program_config()
    assert program_config['qubits'] is not None and len(program_config['qubits']) == 3
    print('Starting toffoli:', program_config['name'])
    setup_data_dir(program_config)
    backend = setup_backend(program_config)
    calibrations = make_single_qutrit_gate_calibrations(backend)
    calibrations = make_qutrit_qubit_cx_calibrations(backend, calibrations)
    runner = setup_runner(backend, calibrations, program_config)
    runner.qutrit_transpile_options.use_waveform = True
    runner.qutrit_transpile_options.remove_custom_pulses = True
    calibrated = load_calibrations(runner, program_config)

    calibrate_single_qutrit_gates(runner, calibrated)
    calibrate_qutrit_qubit_cx(runner)
