import logging
from typing import Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit.providers import Backend

from ..experiment_config import experiments
from ..runners import ExperimentsRunner
from ..util.sizzle import sizzle_hamiltonian_shifts

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner,
    refresh_readout_error: bool = True,
    qutrit_qubit_index: Optional[tuple[int, int]] = None
):
    if qutrit_qubit_index is not None:
        runner_qubits = tuple(runner.qubits)
        runner.qubits = [runner.qubits[idx] for idx in qutrit_qubit_index]

    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    # Find the amplitude that does not disrupt the |2> state too much
    runner.run_experiment('cr_initial_amp')

    # Determine the RCR type to use and corresponding CR width and Rx offset angle
    rough_width_data = runner.run_experiment('cr_rough_width')

    # Eliminate the y component of RCR non-participating state
    runner.run_experiment('cr_angle')

    # Set the target Stark frequency from observed sign of Z
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    fit_params = rough_width_data.analysis_results('simul_fit_params').value[rcr_type]
    omega_z = fit_params[0] * unp.cos(fit_params[2]) / runner.backend.dt
    if abs(omega_z.n) > omega_z.std_dev:
        frequency, amp = get_stark_params(runner.backend, runner.qubits, rcr_type, omega_z.n)
        runner.calibrations.add_parameter_value(frequency, 'stark_frequency', runner.qubits, 'cr')
        # Eliminate the z component of RCR non-participating state
        config = experiments['cr_counter_stark_amp'](runner)
        config.args['amplitudes'] = np.linspace(0., amp * 2., 8)
        runner.run_experiment(config)

    # Calculate the amplitude for CX given the width
    runner.run_experiment('rcr_rough_cr_amp')

    # Minimize the y and z components of RCR
    runner.run_experiment('rcr_rotary_amp')

    # Fine calibration
    runner.run_experiment('crcr_fine_scanbased')

    # Validate CRCR calibration
    runner.run_experiment('crcr_unitaries')

    if qutrit_qubit_index is not None:
        runner.qubits = runner_qubits


def get_stark_params(
    backend: Backend,
    qubits: tuple[int, int],
    control_state: int,
    omega_z: float
) -> float:
    frequencies = nonresonant_frequencies(backend, qubits[1])
    hvars = backend.configuration().hamiltonian['vars']
    amp = 0.08
    shifts_quditbasis = sizzle_hamiltonian_shifts(hvars, qubits, (0., amp), frequencies)[..., 1]
    # shifts is in [Izζ][Iz] basis
    qudit_to_control = np.array([[1., 1., 0.], [1., -1., 1.], [1., 0., -1.]])[control_state]
    shifts = np.einsum('i,fi->f', qudit_to_control, shifts_quditbasis)

    # Align the sign of shifts so we always deal with maxima
    shifts *= -np.sign(omega_z)
    targ = abs(omega_z)
    if np.amax(shifts) < targ:
        # All shifts are smaller than omega_z - increase the amplitude
        return frequencies[np.argmax(shifts)], amp * np.sqrt(targ / np.amax(shifts))
    if not np.any(np.isclose(shifts, targ, atol=5.e+4)):
        # Shifts are either too large or too small - use the minimum above omega_z
        shifts = np.where(shifts > targ, shifts, np.inf)
        return frequencies[np.argmin(shifts)], amp * np.sqrt(targ / np.amin(shifts))
    # Else, return the frequency & amp closest to omega_z
    idx = np.argmin(np.abs(shifts - targ))
    return frequencies[idx], amp


def nonresonant_frequencies(
    backend: Backend,
    qubit: int,
    distance: int = 1,
    npoints: int = 400
) -> np.ndarray:
    """Return disjoint frequency intervals avoiding +-20 MHz resonances of neighboring qubits."""
    neighbors = np.nonzero(backend.coupling_map.distance_matrix[qubit] <= distance)[0]
    qubit_prop = backend.qubit_properties(qubit)
    qubit_props = [backend.qubit_properties(int(q)) for q in neighbors]
    qubit_frequencies = np.array([p.frequency for p in qubit_props])
    anharmonicities = np.array([p.anharmonicity for p in qubit_props])
    resonances = np.sort(np.concatenate([qubit_frequencies, qubit_frequencies + anharmonicities]))
    test_points = np.linspace(qubit_prop.frequency + qubit_prop.anharmonicity + 20.e+6,
                              qubit_prop.frequency + 100.e+6,
                              npoints)
    mask = np.any((test_points[:, None] > resonances[None, :] - 20.e+6)
                  & (test_points[:, None] < resonances[None, :] + 20.e+6), axis=1)
    return test_points[np.logical_not(mask)]
