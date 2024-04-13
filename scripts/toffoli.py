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
    import numpy as np

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)
    
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit_ibm_runtime.exceptions import IBMNotAuthorizedError
    from qutrit_experiments.calibrations import (make_single_qutrit_gate_calibrations,
                                                 make_qutrit_qubit_cx_calibrations,
                                                 make_toffoli_calibrations)
    from qutrit_experiments.configurations.common import (qubits_assignment_error,
                                                          qubits_assignment_error_post)
    import qutrit_experiments.configurations.qutrit_qubit_cx
    import qutrit_experiments.configurations.toffoli
    from qutrit_experiments.experiment_config import ExperimentConfig, ParallelExperimentConfig, register_post
    from qutrit_experiments.experiments.readout_error import CorrelatedReadoutError
    from qutrit_experiments.programs.common import (load_calibrations, setup_backend,
                                                    setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.programs.qutrit_qubit_cx import calibrate_qutrit_qubit_cx, run_unitaries
    from qutrit_experiments.programs.toffoli import characterize_toffoli

    setup_data_dir(program_config)
    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    backend = setup_backend(program_config)
    all_qubits = tuple(program_config['qubits'])

    if len(all_qubits) == 3:
        qubits_assignment_error = 'qubits_assignment_error'

        import qutrit_experiments.configurations.single_qutrit
        qutrit_runner_cls = None
        qutrits = [all_qubits[1]]
    else:
        config = ParallelExperimentConfig(
            run_options={'shots': 10000},
            exp_type='qubits_assignment_error_parallel'
        )
        for ic1 in range(0, len(all_qubits), 3):
            qubits = all_qubits[ic1:ic1 + 3]
            subconf = ExperimentConfig(
                CorrelatedReadoutError,
                qubits,
                exp_type=f'qubits_assignment_error-{"_".join(map(str, qubits))}'
            )
            config.subexperiments.append(subconf)

        qubits_assignment_error = config

        @register_post
        def qubits_assignment_error_parallel(runner, data):
            for ibatch in range(len(all_qubits) // 3):
                qubits_assignment_error_post(runner, data.child_data(ibatch))

        import qutrit_experiments.configurations.full_backend_qutrits
        from qutrit_experiments.runners.parallel_runner import ParallelRunner
        qutrit_runner_cls = ParallelRunner
        qutrits = all_qubits[1::3]

    calibrations = make_single_qutrit_gate_calibrations(backend, qubits=qutrits)
    make_qutrit_qubit_cx_calibrations(backend, calibrations=calibrations, qubits=all_qubits)

    runner = setup_runner(backend, program_config, calibrations=calibrations)
    runner.job_retry_interval = 120

    runner.run_experiment(qubits_assignment_error,
                          force_resubmit=program_config['refresh_readout'])

    calibrated = load_calibrations(runner, program_config)

    qutrit_runner = setup_runner(backend, program_config, calibrations=calibrations,
                                 qubits=qutrits, runner_cls=qutrit_runner_cls)
    qutrit_runner.program_data = runner.program_data
    qutrit_runner.runtime_session = runner.runtime_session
    qutrit_runner.job_retry_interval = 120
    calibrate_single_qutrit_gates(qutrit_runner,
                                  refresh_readout_error=program_config['refresh_readout'],
                                  calibrated=calibrated)

    runner.calibrations = calibrations
    # Session may have been renewed
    runner.runtime_session = qutrit_runner.runtime_session
    
    data_dir = runner.data_dir

    for ic1 in range(0, len(all_qubits), 3):
        toffoli_qubits = all_qubits[ic1:ic1 + 3]
        runner.data_dir = os.path.join(data_dir, '_'.join(map(str, toffoli_qubits)))

        controlling = set(ch['operates']['qubits'][0] for ch in backend.channels.values()
                          if ch['type'] == 'control' and ch['operates']['qubits'][1] == toffoli_qubits[1]) - set(toffoli_qubits)

        if controlling:
            runner.qubits = [toffoli_qubits[1], list(controlling)[0]]
            circuit = QuantumCircuit(2)
            circuit.sx(0)
            circuit.rz(-np.pi / 2., 0)
            circuit.rz(-np.pi / 2., 1)
            circuit.sx(1)
            circuit.rz(-np.pi, 1)
            circuit.ecr(1, 0)
            circuit.rz(-np.pi / 2., 0)
            circuit.sx(0)
            circuit.rz(np.pi / 2., 0)
            circuit.rz(np.pi / 2., 1)
            circuit.sx(1)
            circuit.rz(np.pi / 2., 1)
            run_unitaries(runner, 'reverse_cx_unitaries', circuit=circuit)

        runner.qubits = toffoli_qubits
        calibrate_qutrit_qubit_cx(runner, refresh_readout_error=False, qutrit_qubit_index=(1, 2))
        make_toffoli_calibrations(backend, calibrations, runner.qubits)
        characterize_toffoli(runner, refresh_readout_error=False)
