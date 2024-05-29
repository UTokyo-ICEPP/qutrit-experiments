#!/usr/bin/env python
# flake8: noqa
# pylint: disable=ungrouped-imports, unused-import
"""Experiments to calibrate the qutrit Toffoli gate."""
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == '__main__':
    from qutrit_experiments.script_util.program_config import get_program_config
    program_config = get_program_config()

    try:
        import gpustat
    except ImportError:
        GPU_ID = 0
    else:
        GPU_ID = max(gpustat.new_query(), key=lambda g: g.memory_free).index
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{GPU_ID}'
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    jnp.zeros(1)

    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('qutrit_experiments').setLevel(logging.INFO)

    from qutrit_experiments.calibrations import (
        make_single_qutrit_gate_calibrations,
        make_qutrit_qubit_cx_calibrations,
        make_toffoli_calibrations
    )
    from qutrit_experiments.configurations.common import (
        qubits_assignment_error as _assign_error,
        qubits_assignment_error_post as _assign_post
    )
    import qutrit_experiments.configurations.qutrit_qubit_cx
    import qutrit_experiments.configurations.toffoli
    from qutrit_experiments.experiment_config import (
        ExperimentConfig,
        ParallelExperimentConfig,
        register_exp,
        register_post
    )
    from qutrit_experiments.experiments.readout_error import CorrelatedReadoutError
    from qutrit_experiments.gates import QutritQubitCXType
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.programs.qutrit_qubit_cx import calibrate_qutrit_qubit_cx
    from qutrit_experiments.programs.toffoli import (
        calibrate_toffoli,
        characterize_toffoli,
        characterize_ccz
    )
    from qutrit_experiments.script_util import (
        load_calibrations,
        setup_backend,
        setup_data_dir,
        setup_runner
    )
    from qutrit_experiments.util.qutrit_qubit_cx_type import qutrit_qubit_cx_type

    # Create the data directory
    setup_data_dir(program_config)
    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    # Instantiate the backend
    backend = setup_backend(program_config)
    all_qubits = tuple(program_config['qubits'])

    if len(all_qubits) == 3:
        # Single Toffoli gate calibration
        import qutrit_experiments.configurations.single_qutrit
        qutrit_runner_cls = None  # pylint: disable=invalid-name
        qutrits = [all_qubits[1]]

        # Overwrite qubits_assignment_error registration
        @register_exp
        def qubits_assignment_error(rnr):
            return _assign_error(rnr, all_qubits)

    else:
        # Multiple Toffoli gates calibration (serial)
        import qutrit_experiments.configurations.full_backend_qutrits
        from qutrit_experiments.runners.parallel_runner import ParallelRunner
        qutrit_runner_cls = ParallelRunner  # pylint: disable=invalid-name
        qutrits = all_qubits[1::3]

        @register_exp
        def qubits_assignment_error(_):
            # Readout mitigation in groups of three
            config = ParallelExperimentConfig(
                run_options={'shots': 10000},
                exp_type='qubits_assignment_error_parallel'
            )
            for ictrl1 in range(0, len(all_qubits), 3):
                qubits = all_qubits[ictrl1:ictrl1 + 3]
                subconf = ExperimentConfig(
                    CorrelatedReadoutError,
                    qubits,
                    exp_type=f'qubits_assignment_error-{"_".join(map(str, qubits))}'
                )
                config.subexperiments.append(subconf)

            return config

        @register_post
        def qubits_assignment_error(rnr, data):  # noqa: F811
            for ibatch in range(len(all_qubits) // 3):
                _assign_post(rnr, data.child_data(ibatch))

    calibrations = None
    if program_config['calibrations'] is None:
        # Define all schedules to be calibrated
        calibrations = make_single_qutrit_gate_calibrations(backend, qubits=qutrits)
        make_qutrit_qubit_cx_calibrations(backend, calibrations=calibrations, qubits=all_qubits)
        make_toffoli_calibrations(backend, calibrations=calibrations, qubits=all_qubits)

    # Define the main ExperimentsRunner and run the readout mitigation measurements
    runner = setup_runner(backend, program_config, calibrations=calibrations)
    runner.job_retry_interval = 120
    # Load the calibrations if source is specified in program_config
    calibrated = load_calibrations(runner, program_config)

    try:
        runner.run_experiment('qubits_assignment_error',
                              force_resubmit=program_config['refresh_readout'])

        if qutrit_runner_cls is None:
            qutrit_runner = runner
            qutrit_index = [1]
        else:
            # Define a ParallelRunner to calibrate single qutrit gates in parallel
            qutrit_runner = setup_runner(backend, program_config, calibrations=runner.calibrations,
                                         qubits=qutrits, runner_cls=qutrit_runner_cls)
            qutrit_runner.program_data = runner.program_data
            qutrit_runner.runtime_session = runner.runtime_session
            qutrit_runner.job_retry_interval = 120
            qutrit_index = None  # pylint: disable=invalid-name

        # Single qutrit gates
        calibrate_single_qutrit_gates(qutrit_runner,
                                      refresh_readout_error=program_config['refresh_readout'],
                                      calibrated=calibrated, qutrit_index=qutrit_index)
        # Session may have been renewed
        runner.runtime_session = qutrit_runner.runtime_session

        data_dir = runner.data_dir

        for ic1 in range(0, len(all_qubits), 3):
            # For each qubit combination, make a subdirectory and run the calibrations
            toffoli_qubits = all_qubits[ic1:ic1 + 3]
            runner.data_dir = os.path.join(data_dir, '_'.join(map(str, toffoli_qubits)))
            runner.qubits = toffoli_qubits

            cx_type = qutrit_qubit_cx_type(backend, toffoli_qubits[1:])

            calibrate_qutrit_qubit_cx(runner, cx_type=cx_type, refresh_readout_error=False,
                                      calibrated=calibrated, qutrit_qubit_index=(1, 2))
            calibrate_toffoli(runner, refresh_readout_error=False, calibrated=calibrated)
            if cx_type == QutritQubitCXType.REVERSE:
                characterize_ccz(runner, refresh_readout_error=False)
            else:
                characterize_toffoli(runner, refresh_readout_error=False)

    finally:
        runner.runtime_session.close()
