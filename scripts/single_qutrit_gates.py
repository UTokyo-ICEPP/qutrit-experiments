#!/usr/bin/env python

if __name__ == '__main__':
    import os
    try:
        import gpustat
    except ImportError:
        gpu_id = 0
    else:
        gpu_id = min(gpustat.new_query(), key=lambda g: g.memory_free).index
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qutrit_experiments.calibrations import make_single_qutrit_gate_calibrations
    import qutrit_experiments.configurations.single_qutrit
    from qutrit_experiments.programs.common import (get_program_config, load_calibrations,
                                                    setup_backend, setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import (calibrate_single_qutrit_gates,
                                                                 characterize_qutrit)

    program_config = get_program_config()
    assert program_config['qubits'] is not None and len(program_config['qubits']) == 1
    print('Starting single_qutrit_gates:', program_config['name'])
    setup_data_dir(program_config)
    backend = setup_backend(program_config)
    calibrations = make_single_qutrit_gate_calibrations(backend)
    runner = setup_runner(backend, calibrations, program_config)
    runner.qutrit_transpile_options.use_waveform = True
    runner.qutrit_transpile_options.remove_custom_pulses = True
    calibrated = load_calibrations(runner, program_config)

    runner.program_data['qutrit'] = runner.program_data['qubits'][0]

    calibrate_single_qutrit_gates(runner, calibrated)
    characterize_qutrit(runner)
