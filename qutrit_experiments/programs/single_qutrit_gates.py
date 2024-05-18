from collections.abc import Sequence
import logging
from typing import Optional
from ..runners.experiments_runner import ExperimentsRunner
from .common import run_experiment

logger = logging.getLogger(__name__)


def calibrate_single_qutrit_gates(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    calibrated: Optional[set[str]] = None,
    qutrit_index: Optional[Sequence[int]] = None,
    plot_depth: int = -1,
    save_data: bool = False
):
    if calibrated is None:
        calibrated = set()
    if qutrit_index is not None:
        runner_qubits = list(runner.qubits)
        runner.qubits = [runner.qubits[idx] for idx in qutrit_index]

    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        run_experiment(runner, 'qubits_assignment_error', force_resubmit=refresh_readout_error,
                       plot_depth=plot_depth, save_data=save_data)

    if (exp_type := 'qutrit_wide_frequency') not in calibrated:
        if any(runner.backend.qubit_properties(q).anharmonicity == 0 for q in runner.qubits):
            run_experiment(runner, exp_type, plot_depth=plot_depth, save_data=save_data)

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
            run_experiment(runner, exp_type, plot_depth=plot_depth, save_data=save_data)

    if qutrit_index is not None:
        runner.qubits = runner_qubits


def characterize_qutrit(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    qutrit_index: Optional[Sequence[int]] = None,
    plot_depth: int = -1,
    save_data: bool = False
):
    if qutrit_index is not None:
        runner_qubits = list(runner.qubits)
        runner.qubits = [runner.qubits[idx] for idx in qutrit_index]

    run_experiment(runner, 'qutrit_assignment_error', force_resubmit=refresh_readout_error,
                   plot_depth=plot_depth, save_data=save_data)

    run_experiment(runner, 'qutrit_t1', plot_depth=plot_depth, save_data=save_data)

    runner.qutrit_transpile_options.rz_casted_gates = 'all'
    run_experiment(runner, 'qutrit_x12_irb', plot_depth=plot_depth, save_data=save_data)

    if qutrit_index is not None:
        runner.qubits = runner_qubits
