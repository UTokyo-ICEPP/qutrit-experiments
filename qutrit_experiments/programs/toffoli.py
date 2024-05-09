from itertools import product
from typing import Optional
import numpy as np
from qiskit_experiments.data_processing import DataProcessor

from ..data_processing import ReadoutMitigation
from ..experiments.circuit_runner import CircuitRunner
from ..experiment_config import ExperimentConfig
from ..gates import QutritQubitCXType
from ..runners import ExperimentsRunner


def calibrate_toffoli(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    calibrated: Optional[set[str]] = None
):
    if calibrated is None:
        calibrated = set()
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    if (exp_type := 'c1c2_cr_rotary_delta') not in calibrated:
        runner.run_experiment(exp_type)

    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits[1:])
    if rcr_type == QutritQubitCXType.REVERSE:
        for exp_type in ['cz_c2_phase', 'ccz_c2_phase']:
            if exp_type not in calibrated:
                runner.run_experiment(exp_type)


def characterize_ccz(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Truth table
    runner.run_experiment('truthtable_ccz', analyze=False)
    runner.run_experiment('phasetable_ccz', analyze=False)
    runner.run_experiment('xminusxplus_c2_phase', analyze=False)
    qpt_data = runner.run_experiment('qpt_ccz_bc', analyze=False)

    #recover_qpt_data(runner, qpt_data)


def characterize_toffoli(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True
):
    if 'readout_mitigator' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Truth table
    runner.run_experiment('truthtable_toffoli')


def ccz_qpt_expected():
    init_1q = np.array([
        [1., 0.],
        [0., 1.],
        [1. / np.sqrt(2.), 1. / np.sqrt(2.)],
        [1. / np.sqrt(2.), 1.j / np.sqrt(2.)]
    ], dtype=complex)
    inits = np.empty((64, 8), dtype=complex)
    for ip, p_idx in enumerate(product(range(4), range(4), range(4))):
        inits[ip] = np.kron(init_1q[p_idx[2]], np.kron(init_1q[p_idx[1]], init_1q[p_idx[0]]))

    meas_1q = np.empty((3, 2, 2), dtype=complex)
    meas_1q[0] = np.eye(2, dtype=complex)
    meas_1q[1] = np.array([[1., 1.], [1., -1.]], dtype=complex) / np.sqrt(2.)
    meas_1q[2] = np.array([[1., -1.j], [1., 1.j]]) / np.sqrt(2.)
    bases = np.empty((27, 8, 8), dtype=complex)
    for ib, m_idx in enumerate(product(range(3), range(3), range(3))):
        bases[ib] = np.kron(meas_1q[m_idx[2]], np.kron(meas_1q[m_idx[1]], meas_1q[m_idx[0]]))

    ccz = np.diagflat([1.] * 7 + [-1.])
    states = np.einsum('imj,jk,lk->lim', bases, ccz, inits).reshape(-1, 8)
    return np.square(np.abs(states))


def recover_qpt_data(runner, qpt_data):
    # Resubmit circuits with bad measurements
    matrix = runner.program_data['readout_mitigator'][runner.qubits].assignment_matrix(runner.qubits)
    dp = DataProcessor('counts', [ReadoutMitigation(matrix)])

    obss = []
    for proc_data in dp(qpt_data.data()):
        obs = np.zeros(8)
        for key, cnt in proc_data.items():
            obs[int(key, 2)] = cnt
        obss.append(obs / np.sum(obs))
    obss = np.array(obss)

    exps = ccz_qpt_expected()

    sqdiffs = np.sqrt(np.sum(np.square(exps - obss), axis=1))
    mean = np.mean(sqdiffs)
    stddev = np.std(sqdiffs)
    anomalous = np.nonzero(np.abs(sqdiffs - mean) > 5. * stddev)[0]

    cc = runner.make_experiment('qpt_ccz_bc').circuits()
    circuits = [cc[ic] for ic in anomalous]
    config = ExperimentConfig(
        CircuitRunner,
        runner.qubits,
        args={'circuits': circuits},
        run_options={'shots': 2000},
        exp_type='ccz_qpt_recover'
    )
    return runner.run_experiment(config)
