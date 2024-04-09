#!/usr/bin/env python

if __name__ == '__main__':
    import os
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
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import logging
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError
    from qutrit_experiments.calibrations import (make_single_qutrit_gate_calibrations,
                                                 make_qutrit_qubit_cx_calibrations,
                                                 make_toffoli_calibrations)
    import qutrit_experiments.configurations.qutrit_qubit_cx
    import qutrit_experiments.configurations.toffoli
    from qutrit_experiments.programs.common import (get_program_config, load_calibrations,
                                                    setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.programs.qutrit_qubit_cx import calibrate_qutrit_qubit_cx
    from qutrit_experiments.programs.toffoli import characterize_toffoli

    program_config = get_program_config()
    setup_data_dir(program_config)
    while True:
        try:
            service = QiskitRuntimeService(channel='ibm_quantum', instance=program_config['instance'])
            backend = service.backend(program_config['backend'], instance=program_config['instance'])
        except IBMNotAuthorizedError:
            continue
        break

    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    all_qubits = tuple(program_config['qubits'])



    runner = setup_runner(backend, program_config)
    runner.job_retry_interval = 120
    runner.run_experiment('qubits_assignment_error',
                          force_resubmit=program_config['refresh_readout'])



    if len(all_qubits) == 3:
        import qutrit_experiments.configurations.single_qutrit
        qutrit_runner_cls = None
        qutrits = [all_qubits[1]]
    else:
        import qutrit_experiments.configurations.full_backend_qutrits
        from qutrit_experiments.runners.parallel_runner import ParallelRunner
        qutrit_runner_cls = ParallelRunner
        qutrits = all_qubits[1::3]

    
    qutrit_runner = setup_runner(backend, program_config, calibrations=calibrations,
                                 qubits=qutrits, runner_cls=qutrit_runner_cls)
    qutrit_runner.program_data = runner.program_data
    qutrit_runner.runtime_session = runner.runtime_session
    qutrit_runner.job_retry_interval = 120
    calibrated = load_calibrations(qutrit_runner, program_config)
    calibrate_single_qutrit_gates(qutrit_runner,
                                  refresh_readout_error=program_config['refresh_readout'],
                                  calibrated=calibrated, qutrit_index=[1])
    
    # Session may have been renewed
    runner.runtime_session = qutrit_runner.runtime_session
    calibrations = make_qutrit_qubit_cx_calibrations(backend, calibrations, qubits=all_qubits)

    for icomb in range(len(all_qubits) // 3):
        runner.qubits = all_qubits[icomb * 3:(icomb + 1) * 3]
        calibrate_qutrit_qubit_cx(runner, refresh_readout_error=False, qutrit_qubit_index=(1, 2))
        make_toffoli_calibrations(backend, calibrations, runner.qubits)
        characterize_toffoli(runner, refresh_readout_error=False)
