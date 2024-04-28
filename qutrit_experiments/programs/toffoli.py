from ..runners import ExperimentsRunner


def calibrate_toffoli(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    runner.run_experiment('c1c2_rotary_delta')


def characterize_ccz(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Truth table
    runner.run_experiment('ccz_truth_table')
    runner.run_experiment('ccz_phase_table')
    runner.run_experiment('ccz_qpt_bc')


def characterize_toffoli(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Truth table
    runner.run_experiment('toffoli_truth_table')
