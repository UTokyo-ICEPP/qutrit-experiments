#!/usr/bin/env python
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging

if __name__ == '__main__':
    from qutrit_experiments.programs.program_config import get_program_config
    program_config = get_program_config()

    try:
        import gpustat
    except ImportError:
        gpu_id = 0
    else:
        gpu_id = max(gpustat.new_query(), key=lambda g: g.memory_free).index
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    jnp.zeros(1)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qutrit_experiments.calibrations import make_single_qutrit_gate_calibrations
    from qutrit_experiments.constants import RESTLESS_REP_DELAY
    from qutrit_experiments.programs.common import (load_calibrations, setup_backend,
                                                    setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import (calibrate_single_qutrit_gates,
                                                                 characterize_qutrit)

    program_config = get_program_config()
    assert program_config['qubits'] is not None
    print('Starting single_qutrit_gates:', program_config['name'])

    if (nq := len(program_config['qubits'])) == 1:
        import qutrit_experiments.configurations.single_qutrit
        runner_cls = None
    else:
        import qutrit_experiments.configurations.full_backend_qutrits
        from qutrit_experiments.runners import ParallelRunner
        runner_cls = ParallelRunner

    setup_data_dir(program_config)
    backend = setup_backend(program_config)
    calibrations = make_single_qutrit_gate_calibrations(backend)

    qubits = []
    props = backend.properties()
    for qubit in program_config['qubits']:
        if props.qubit_property(qubit).get('T1', (0.,))[0] > RESTLESS_REP_DELAY:
            qubits.append(qubit)
    if not qubits:
        raise RuntimeError('No qubits have T1 > RESTLESS_REP_DELAY')
    program_config['qubits'] = qubits

    runner = setup_runner(backend, program_config, calibrations=calibrations, runner_cls=runner_cls)
    runner.job_retry_interval = 120
    runner.default_print_level = 1
    calibrated = load_calibrations(runner, program_config)

    try:
        calibrate_single_qutrit_gates(runner, refresh_readout_error=program_config['refresh_readout'],
                                      calibrated=calibrated)
        characterize_qutrit(runner, refresh_readout_error=program_config['refresh_readout'])
    finally:
        runner.runtime_session.close()
