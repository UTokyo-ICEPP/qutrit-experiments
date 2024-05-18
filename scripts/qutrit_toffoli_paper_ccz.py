#!/usr/bin/env python
"""Experiments characterizing the CCZ sequence for the qutrit Toffoli paper."""
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

    from qiskit.circuit import Parameter
    from qutrit_experiments.calibrations import (make_single_qutrit_gate_calibrations,
                                                 make_qutrit_qubit_cx_calibrations,
                                                 make_toffoli_calibrations)
    from qutrit_experiments.configurations.common import (qubits_assignment_error as _assign_error,
                                                          qubits_assignment_error_post as _assign_post)
    import qutrit_experiments.configurations.full_backend_qutrits
    import qutrit_experiments.configurations.qutrit_qubit_cx
    import qutrit_experiments.configurations.toffoli
    import qutrit_experiments.configurations.qutrit_toffoli_paper
    from qutrit_experiments.experiment_config import (BatchExperimentConfig,
                                                      ParallelExperimentConfig,
                                                      experiments, postexperiments)
    from qutrit_experiments.gates import QutritQubitCXType
    from qutrit_experiments.programs.common import (load_calibrations, run_experiment, setup_backend,
                                                    setup_data_dir, setup_runner)
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.runners.parallel_runner import ParallelRunner


    # Create the data directory
    setup_data_dir(program_config)
    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    # Instantiate the backend
    backend = setup_backend(program_config)
    all_qubits = tuple(program_config['qubits'])
    qubits_list = [all_qubits[i:i + 3] for i in range(0, len(all_qubits), 3)]
    qutrits = [qubits[1] for qubits in qubits_list]

    # Define all schedules to be calibrated
    calibrations = make_single_qutrit_gate_calibrations(backend, qubits=qutrits)
    make_qutrit_qubit_cx_calibrations(backend, calibrations=calibrations, qubits=all_qubits)
    make_toffoli_calibrations(backend, calibrations=calibrations, qubits=all_qubits)
    calibrations._register_parameter(Parameter('delta_cz_id'), ())
    calibrations._register_parameter(Parameter('delta_ccz_id0'), ())
    calibrations._register_parameter(Parameter('delta_ccz_id1'), ())
    calibrations._register_parameter(Parameter('delta_ccz_cz'), ())

    for qubits in qubits_list:
        calibrations.add_parameter_value(int(QutritQubitCXType.REVERSE), 'rcr_type', qubits[1:])

    # Define the main ExperimentsRunner and run the readout mitigation measurements
    runner = setup_runner(backend, program_config, calibrations=calibrations)
    runner.job_retry_interval = 120
    runner.default_print_level = 1

    def parallelized(exp_type, gen=None, qidx=None):
        config = ParallelExperimentConfig(exp_type=exp_type)
        runner_qubits = runner.qubits
        for qubits in qubits_list:
            if qidx:
                runner.qubits = [qubits[iq] for iq in qidx]
            else:
                runner.qubits = qubits
            if gen:
                subconfig = gen(runner)
            else:
                subconfig = experiments[exp_type](runner)
            config.subexperiments.append(subconfig)

        runner.qubits = runner_qubits
        return config

    try:
        config = parallelized('qubits_assignment_error',
                              gen=lambda runner: _assign_error(runner, runner.qubits))
        config.run_options = {'shots': 10000}
        postexperiments.pop('qubits_assignment_error')
        data = run_experiment(runner, config, plot_depth=-1,
                              force_resubmit=program_config['refresh_readout'])
        for qubits, child_data in zip(qubits_list, data.child_data()):
            runner.qubits = qubits
            _assign_post(runner, child_data)

        # Load the calibrations if source is specified in program_config
        calibrated = load_calibrations(runner, program_config)

        # Define a ParallelRunner to calibrate single qutrit gates in parallel
        qutrit_runner = setup_runner(backend, program_config, calibrations=calibrations,
                                     qubits=qutrits, runner_cls=ParallelRunner)
        qutrit_runner.program_data = runner.program_data
        qutrit_runner.runtime_session = runner.runtime_session
        qutrit_runner.job_retry_interval = 120
        qutrit_runner.default_print_level = 0

        # Single qutrit gates
        calibrate_single_qutrit_gates(qutrit_runner,
                                      refresh_readout_error=program_config['refresh_readout'],
                                      calibrated=calibrated)
        # Session may have been renewed
        runner.runtime_session = qutrit_runner.runtime_session

        # Update the qubits list to exclude combinations with bad qutrits
        good_qutrits = set(qutrit_runner.qubits)
        qubits_list = [all_qubits[i:i + 3] for i in range(0, len(all_qubits), 3)
                       if all_qubits[i + 1] in good_qutrits]

        if (exp_type := 'tc2_cr_rotary_delta') not in calibrated:
            config = parallelized(exp_type, qidx=(1, 2))
            run_experiment(runner, config, plot_depth=-1)

        if (exp_type := 'c1c2_cr_rotary_delta') not in calibrated:
            config = parallelized(exp_type)
            run_experiment(runner, config, plot_depth=-1)

        batch_config = BatchExperimentConfig(exp_type='delta_cz_cal')
        for exp_type in ['cz_c2_phase', 'cz_id_c2_phase']:
            if exp_type not in calibrated:
                config = parallelized(exp_type)
                batch_config.subexperiments.append(config)
        if batch_config.subexperiments:
            run_experiment(runner, batch_config, plot_depth=-1)

        batch_config = BatchExperimentConfig(exp_type='delta_ccz_cal')
        for exp_type in [
            'ccz_c2_phase',
            'ccz_id0_c2_phase',
            'ccz_id1_c2_phase',
            'ccz_cz_c2_phase'
        ]:
            if exp_type not in calibrated:
                config = parallelized(exp_type)
                batch_config.subexperiments.append(config)
        if batch_config.subexperiments:
            run_experiment(runner, batch_config, plot_depth=-1)

        config = parallelized('qpt_ccz_bc')
        config.experiment_options = {'max_circuits': 100}
        config.run_options = {'shots': 2000}
        config.analysis_options = {'parallelize': 0}
        data_qpt = run_experiment(runner, config, block_for_results=False, plot_depth=-1)

        batch_config = BatchExperimentConfig(
            exp_type='characterization_3q',
            experiment_options={'max_circuits': 100},
            run_options={'shots': 2000},
        )
        for exp_type in [
            'truthtable_ccz',
            'phasetable_ccz',
            'truthtable_id0',
            'phasetable_id0',
            'truthtable_id1',
            'phasetable_id1',
            'truthtable_cz',
            'phasetable_cz'
        ]:
            config = parallelized(exp_type)
            batch_config.subexperiments.append(config)
        data_3q = run_experiment(runner, batch_config, block_for_results=False, plot_depth=-1)

        batch_config = BatchExperimentConfig(
            exp_type='characterization_1q',
            run_options={'shots': 2000}
        )
        for exp_type in [
            'c2phase_xminusxplus',
            'qpt_xminusxplus',
            'c2phase_xplus3',
            'qpt_xplus3'
        ]:
            config = parallelized(exp_type)
            batch_config.subexperiments.append(config)
        data_1q = run_experiment(runner, batch_config, block_for_results=False, plot_depth=-1)

    finally:
        runner.runtime_session.close()

    data_qpt.block_for_results()
    data_3q.block_for_results()
    data_1q.block_for_results()

    runner.program_data['choi'] = {}
    runner.program_data['process_fidelity'] = {}
    for child_data in data_qpt.child_data():
        qubits = tuple(child_data.metadata['physical_qubits'])
        runner.program_data['choi'][qubits] = child_data.analysis_results('state').value
        runner.program_data['process_fidelity'][qubits] = child_data.analysis_results('process_fidelity').value

    for pdata in data_3q.child_data():
        exp_type = pdata.experiment_type
        runner.program_data[exp_type] = {}
        for child_data in pdata.child_data():
            qubits = tuple(child_data.metadata['physical_qubits'])
            if exp_type.startswith('truthtable'):
                runner.program_data[exp_type][qubits] = child_data.analysis_results('truth_table').value
            else:
                runner.program_data[exp_type][qubits] = child_data.analysis_results('phases').value

    for pdata in data_1q.child_data():
        exp_type = pdata.experiment_type
        runner.program_data[exp_type] = {}
        for child_data in pdata.child_data():
            qubits = tuple(child_data.metadata['physical_qubits'])
            if exp_type.startswith('c2phase'):
                runner.program_data[exp_type][qubits] = child_data.analysis_results('phase_offset').value
            else:
                runner.program_data[exp_type][qubits] = child_data.analysis_results('process_fidelity').value
