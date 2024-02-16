from typing import Optional
from ..runners.experiments_runner import ExperimentsRunner
from ..configurations import single_qutrit


def calibrate_single_qutrit_gates(runner: ExperimentsRunner, calibrated: Optional[set[str]] = None):
    if calibrated is None:
        calibrated = set()

    runner.program_data['qutrit'] = runner.program_data['qubits'][0]
    exp_types = [
        'qutrit_rough_frequency',
        'qutrit_rough_amplitude',
        'qubit_assignment_error',
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


def characterize_qutrit(runner: ExperimentsRunner):
    runner.program_data['qutrit'] = runner.program_data['qubits'][0]
    exp_types = [
        'qutrit_assignment_error',
        'qutrit_t1',
        'qutrit_x12_irb'
    ]
    for exp_type in exp_types:
        runner.run_experiment(exp_type)
