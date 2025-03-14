#!/usr/bin/env python
# flake8: noqa
# pylint: disable=ungrouped-imports, unused-import
"""Experiments characterizing the CCZ sequence for the qutrit Toffoli paper."""
import os
import sys
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


if __name__ == '__main__':
    from qutrit_experiments.script_util.program_config import get_program_config
    program_config = get_program_config(
        additional_args=[
            (('--no-cal',),
             {'help': 'Do not perform any calibration experiment.', 'action': 'store_true',
              'dest': 'no_cal'}),
            (('--no-qpt',),
             {'help': 'Skip the QPT experiment.', 'action': 'store_true', 'dest': 'no_qpt'}),
            (('--no-3q',),
             {'help': 'Skip 3Q characterization experiments.',
              'action': 'store_true', 'dest': 'no_3q'}),
            (('--no-1q',),
             {'help': 'Skip 1Q characterization experiments.',
              'action': 'store_true', 'dest': 'no_1q'})
        ]
    )

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
    LOG = logging.getLogger(__name__)
    LOG.setLevel(logging.INFO)

    from qiskit.circuit import Parameter
    from qiskit_ibm_runtime import Batch, Session
    from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment
    from qutrit_experiments.calibrations import (
        make_single_qutrit_gate_calibrations,
        make_qutrit_qubit_cx_calibrations,
        make_toffoli_calibrations
    )
    from qutrit_experiments.configurations.common import (
        configure_qpt_readout_mitigation,
        configure_readout_mitigation,
        qubits_assignment_error as _assign_error,
        qubits_assignment_error_post as _assign_post
    )
    import qutrit_experiments.configurations.full_backend_qutrits
    import qutrit_experiments.configurations.qutrit_qubit_cx
    import qutrit_experiments.configurations.toffoli
    import qutrit_experiments.configurations.qutrit_toffoli_paper
    from qutrit_experiments.experiment_config import (
        BatchExperimentConfig,
        CompositeExperimentConfig,
        ParallelExperimentConfig,
        experiments,
        postexperiments
    )
    from qutrit_experiments.experiments.truth_table import TruthTable
    from qutrit_experiments.gates import QutritQubitCXType
    from qutrit_experiments.programs.common import run_experiment
    from qutrit_experiments.programs.single_qutrit_gates import calibrate_single_qutrit_gates
    from qutrit_experiments.runners.parallel_runner import ParallelRunner
    from qutrit_experiments.script_util import (
        load_calibrations,
        setup_backend,
        setup_data_dir,
        setup_runner
    )

    # Create the data directory
    setup_data_dir(program_config)
    assert program_config['qubits'] is not None and len(program_config['qubits']) % 3 == 0
    print('Starting toffoli:', program_config['name'])

    # Instantiate the backend
    backend = setup_backend(program_config)
    all_qubits = tuple(program_config['qubits'])
    qubits_list = [all_qubits[i:i + 3] for i in range(0, len(all_qubits), 3)]
    qutrits = [qubits[1] for qubits in qubits_list]

    calibrations = None
    if program_config['calibrations'] is None:
        # Define all schedules to be calibrated
        calibrations = make_single_qutrit_gate_calibrations(backend, qubits=qutrits)
        make_qutrit_qubit_cx_calibrations(backend, calibrations=calibrations, qubits=all_qubits)
        make_toffoli_calibrations(backend, calibrations=calibrations, qubits=all_qubits)
        calibrations._register_parameter(Parameter('delta_cz_id'), ())
        calibrations._register_parameter(Parameter('delta_ccz_id0'), ())
        calibrations._register_parameter(Parameter('delta_ccz_id1'), ())
        calibrations._register_parameter(Parameter('delta_ccz_id2'), ())

        for qubits in qubits_list:
            calibrations.add_parameter_value(int(QutritQubitCXType.REVERSE), 'rcr_type', qubits[1:])

    # Define the main ExperimentsRunner and run the readout mitigation measurements
    runner = setup_runner(backend, program_config, calibrations=calibrations)
    runner.job_retry_interval = 120
    runner.default_print_level = 1

    # Load the calibrations if source is specified in program_config
    calibrated = load_calibrations(runner, program_config)

    def parallelized(typ, genfn=None, qidx=None):
        """Return a ParallelExperimentConfig."""
        cfg = ParallelExperimentConfig(exp_type=typ, flatten_results=False)
        runner_qubits = runner.qubits
        for pqs in qubits_list:
            if qidx:
                runner.qubits = [pqs[iq] for iq in qidx]
            else:
                runner.qubits = pqs
            if genfn:
                subcfg = genfn(runner)
            else:
                subcfg = experiments[typ](runner)
            cfg.subexperiments.append(subcfg)

        runner.qubits = runner_qubits
        return cfg

    rem_config = parallelized('qubits_assignment_error',
                                genfn=lambda runner: _assign_error(runner, runner.qubits))
    rem_config.run_options = {'shots': 10000}
    postexperiments.pop('qubits_assignment_error')

    if not program_config['no_cal']:
        # Define a ParallelRunner to calibrate single qutrit gates in parallel
        qutrit_runner = setup_runner(backend, program_config, calibrations=calibrations,
                                        qubits=qutrits, runner_cls=ParallelRunner)
        qutrit_runner.job_retry_interval = 120
        qutrit_runner.default_print_level = 0

        with Session(backend=backend):
            data = run_experiment(runner, rem_config, plot_depth=-1,
                                  force_resubmit=program_config['refresh_readout'])
            for physical_qubits, child_data in zip(qubits_list, data.child_data()):
                runner.qubits = physical_qubits
                _assign_post(runner, child_data)

            qutrit_runner.program_data = runner.program_data

            # Single qutrit gates
            calibrate_single_qutrit_gates(
                qutrit_runner,
                refresh_readout_error=program_config['refresh_readout'],
                calibrated=calibrated
            )

            # Update the qubits list to exclude combinations with bad qutrits
            good_qutrits = set(qutrit_runner.qubits)
            qubits_list = [all_qubits[i:i + 3] for i in range(0, len(all_qubits), 3)
                            if all_qubits[i + 1] in good_qutrits]

            if (exp_type := 'tc2_cr_rotary_delta') not in calibrated:
                run_experiment(runner, parallelized(exp_type, qidx=(1, 2)), plot_depth=-1)

            if (exp_type := 'c1c2_cr_rotary_delta') not in calibrated:
                run_experiment(runner, parallelized(exp_type), plot_depth=-1)

            config = BatchExperimentConfig(exp_type='delta_cz_cal')
            for exp_type in ['cz_c2_phase', 'cz_id_c2_phase']:
                if exp_type not in calibrated:
                    config.subexperiments.append(parallelized(exp_type))
            if config.subexperiments:
                run_experiment(runner, config, plot_depth=-1)

            config = BatchExperimentConfig(exp_type='delta_ccz_cal')
            for exp_type in [
                'ccz_c2_phase',
                'ccz_id0_c2_phase',
                'ccz_id1_c2_phase',
                'ccz_id2_c2_phase'
            ]:
                if exp_type not in calibrated:
                    config.subexperiments.append(parallelized(exp_type))
            if config.subexperiments:
                run_experiment(runner, config, plot_depth=-1)

    configs = []
    exp_data = {}

    with Batch(backend=backend):
        rem_config.exp_type += '_char'
        exp_data[rem_config.exp_type] = run_experiment(
            runner, rem_config, plot_depth=-1, force_resubmit=program_config['refresh_readout']
        )

        if not program_config['no_qpt']:
            def gen(rnr):
                cfg = experiments['qpt_ccz_bc'](rnr)
                cfg.analysis_options.pop('target_bootstrap_samples')
                return cfg

            config = parallelized('qpt_ccz_bc', genfn=gen)
            config.experiment_options = {'max_circuits': 100}
            config.run_options = {'shots': 2000}

            configs.append(config)
            exp_data[config.exp_type] = run_experiment(runner, config, analyze=False,
                                                        block_for_results=False)

        if not program_config['no_3q']:
            for seq_name in ['ccz', 'id0', 'id1', 'id2', 'id3']:
                config = BatchExperimentConfig(
                    exp_type=f'characterization_{seq_name}',
                    experiment_options={'max_circuits': 100},
                    run_options={'shots': 2000},
                )
                for etype in ['truthtable', 'phasetable']:
                    config.subexperiments.append(parallelized(f'{etype}_{seq_name}'))

                configs.append(config)
                exp_data[config.exp_type] = run_experiment(runner, config, analyze=False,
                                                            block_for_results=False)

        if not program_config['no_1q']:
            config = BatchExperimentConfig(
                exp_type='characterization_1q',
                run_options={'shots': 2000}
            )
            for exp_type in [
                'c2phase_xminusxplus',
                'qpt_xminusxplus',
                'c2phase_xplus3',
                'qpt_xplus3'
            ]:
                config.subexperiments.append(parallelized(exp_type))

            configs.append(config)
            exp_data[config.exp_type] = run_experiment(runner, config, analyze=False,
                                                        block_for_results=False)

    exp_data[rem_config.exp_type].block_for_results()
    for physical_qubits, child_data in zip(qubits_list, exp_data[rem_config.exp_type].child_data()):
        runner.qubits = physical_qubits
        _assign_post(runner, child_data)

    def set_rem(cfg):
        if isinstance(cfg, CompositeExperimentConfig):
            for subcfg in cfg.subexperiments:
                set_rem(subcfg)
            return
        runner.qubits = next(qubits for qubits in qubits_list
                             if set(cfg.physical_qubits) <= set(qubits))
        if issubclass(cfg.cls, TomographyExperiment):
            configure_qpt_readout_mitigation(runner, cfg)
        else:
            configure_readout_mitigation(runner, cfg,
                                         probability=not issubclass(cfg.cls, TruthTable))

    for config in configs:
        set_rem(config)
        exp = runner.make_experiment(config)
        LOG.info('Analyzing %s', config.exp_type)
        exp.analysis.run(exp_data[config.exp_type]).block_for_results()

    def save_figure(chd):
        typ = chd.experiment_type
        pqs = chd.metadata['physical_qubits']
        for name in chd.figure_names:
            file_name = os.path.join(runner.data_dir,
                                     'program_data',
                                     f'{typ}_{"_".join(map(str, pqs))}_{name}')
            figure = chd.figure(name).figure
            figure.savefig(file_name + '.pdf')
            figure.savefig(file_name + '.jpg')

    if (data_qpt := exp_data.get('qpt_ccz_bc')) is not None:
        runner.program_data['choi'] = {}
        runner.program_data['process_fidelity'] = {}
        for child_data in data_qpt.child_data():
            qubits = tuple(child_data.metadata['physical_qubits'])
            runner.program_data['choi'][qubits] = child_data.analysis_results('state').value
            runner.program_data['process_fidelity'][qubits] = \
                child_data.analysis_results('process_fidelity').value

    for seq_name in ['ccz', 'id0', 'id1', 'id2', 'id3']:
        if (bdata := exp_data.get(f'characterization_{seq_name}')) is None:
            continue

        for (pdata, etype, resname) in zip(bdata.child_data(),
                                           ['truthtable', 'phasetable'],
                                           ['truth_table', 'phases']):
            exp_type = pdata.experiment_type
            runner.program_data[exp_type] = {}
            for child_data in pdata.child_data():
                qubits = tuple(child_data.metadata['physical_qubits'])
                runner.program_data[exp_type][qubits] = child_data.analysis_results(resname).value
                save_figure(child_data)

    if (bdata := exp_data.get('characterization_1q')) is not None:
        for pdata in bdata.child_data():
            exp_type = pdata.experiment_type
            runner.program_data[exp_type] = {}
            for child_data in pdata.child_data():
                qubits = tuple(child_data.metadata['physical_qubits'])
                if exp_type.startswith('c2phase'):
                    runner.program_data[exp_type][qubits] = \
                        child_data.analysis_results('phase_offset').value
                else:
                    runner.program_data[exp_type][qubits] = \
                        child_data.analysis_results('process_fidelity').value
                save_figure(child_data)

    runner.save_program_data()
