from ..runners import ExperimentsRunner

def characterize_toffoli(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Truth table
    runner.run_experiment('toffoli_truth_table')
