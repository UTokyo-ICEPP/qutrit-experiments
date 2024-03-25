from typing import Optional
from ..runners.experiments_runner import ExperimentsRunner


def calibrate_single_qutrit_gates(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    calibrated: Optional[set[str]] = None
):
    if calibrated is None:
        calibrated = set()

    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    active_qubits = set(runner.qubits)

    exp_types = [
        'qutrit_rough_frequency',
        'qutrit_rough_amplitude',
        'qutrit_semifine_frequency',
        'qutrit_fine_frequency',
        'qutrit_rough_x_drag',
        'qutrit_rough_sx_drag',
        'qutrit_fine_sx_amplitude',
        'qutrit_fine_sx_drag',
        'qutrit_fine_x_amplitude',
        'qutrit_fine_x_drag',
        'qutrit_x12_stark_shift',
        'qutrit_sx12_stark_shift',
        'qutrit_x_stark_shift',
        'qutrit_sx_stark_shift'
    ]
    for exp_type in exp_types:
        if exp_type not in calibrated:
            runner.run_experiment(exp_type)
            # Exclude qubits that failed calibration
            cal_data = runner.calibrations.parameters_table(group=exp_type,
                                                            most_recent_only=False)['data']
            active_qubits &= set(row['qubits'][0] for row in cal_data)
            runner.qubits = list(active_qubits)


def characterize_qutrit(runner: ExperimentsRunner):
    for exp_type in ['qutrit_assignment_error', 'qutrit_t1']:
        runner.run_experiment(exp_type)

    runner.qutrit_transpile_options.rz_casted_gates = 'all'
    runner.run_experiment('qutrit_x12_irb')
