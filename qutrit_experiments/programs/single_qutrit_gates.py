import logging
from typing import Optional
from ..runners.experiments_runner import ExperimentsRunner

logger = logging.getLogger(__name__)


def _run_experiment(runner, exp_type, is_calibration=True, force_resubmit=False):
    runner_qubits = set(runner.qubits)

    exp_data = None
    if runner.saved_data_exists(exp_type):
        exp_data = runner.load_data(exp_type)
        if (data_qubits := set(exp_data.metadata['physical_qubits'])) - runner_qubits:
            logger.warning('Saved experiment data for %s has out-of-configuration qubits.',
                            exp_type)
        runner.qubits = data_qubits

    runner.run_experiment(exp_type, exp_data=exp_data, force_resubmit=force_resubmit)

    if is_calibration:
        # Exclude qubits that failed calibration
        cal_data = runner.calibrations.parameters_table(group=exp_type,
                                                        most_recent_only=False)['data']
        runner.qubits = runner_qubits & set(row['qubits'][0] for row in cal_data)
    else:
        runner.qubits = runner_qubits

    return exp_data


def calibrate_single_qutrit_gates(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    calibrated: Optional[set[str]] = None
):
    if calibrated is None:
        calibrated = set()

    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        _run_experiment(runner, 'qubits_assignment_error', is_calibration=False,
                        force_resubmit=refresh_readout_error)

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
            _run_experiment(runner, exp_type)


def characterize_qutrit(runner: ExperimentsRunner):
    for exp_type in ['qutrit_assignment_error', 'qutrit_t1']:
        _run_experiment(runner, exp_type, is_calibration=False)

    runner.qutrit_transpile_options.rz_casted_gates = 'all'
    _run_experiment(runner, 'qutrit_x12_irb', is_calibration=False)
