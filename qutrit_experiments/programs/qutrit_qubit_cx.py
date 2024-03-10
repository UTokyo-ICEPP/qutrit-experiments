import logging
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit_experiments.framework import ExperimentData

from ..experiment_config import experiments
from ..experiments.qutrit_qubit_cx.rotary import rotary_angle_per_amp
from ..experiments.qutrit_qubit_cx.util import RCRType
from ..runners import ExperimentsRunner
from ..util.bloch import so3_cartesian, so3_cartesian_params
from ..util.sizzle import sizzle_hamiltonian_shifts

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner,
    initial_cr_amp: float = 0.8,
    refresh_readout_error: bool = True
):
    qubits = runner.program_data['qubits'][1:]

    if 'readout_assignment_matrices' not in runner.program_data:
        # Construct the error mitigation matrix and find the rough CR pulse width
        runner.run_experiment('qubits_assignment_error', force_resubmit=refresh_readout_error)

    runner.calibrations.add_parameter_value(initial_cr_amp, 'cr_amp', qubits, 'cr')
    rough_width_data = runner.run_experiment('c2t_cr_rough_width')

    # Check the Z components of the CR Hamiltonian and decide whether to tune down the amplitude or
    # introduce siZZle
    # Add rotary based on a rough guess
    setup_cr(rough_width_data, runner)

    runner.run_experiment('c2t_crcr_cr_width')

    config = experiments['c2t_cr_unitaries'](runner)
    config.exp_type += '_postwidth'
    data = runner.run_experiment(config)
    unitary_params = data.analysis_results('unitary_parameters').value
    cr_params = np.array([unp.nominal_values(unitary_params[ic]) for ic in range(3)])

    rotary_angle = get_rotary_guess(cr_params, runner)
    runner.program_data['crcr_rotary_test_angles'] = np.linspace(
        rotary_angle - np.pi / 6., rotary_angle + np.pi / 6., 6
    )

    for exp_type in [
        'c2t_crcr_rotary',
        'c2t_crcr_angle_width_rate',
        'c2t_crcr_fine_iter1',
        'c2t_crcr_fine_iter2',
        'c2t_crcr_unitaries'
    ]:
        runner.run_experiment(exp_type)


def setup_cr(rough_width_data: ExperimentData, runner: ExperimentsRunner):
    qubits = tuple(runner.program_data['qubits'][1:])

    fit_params = rough_width_data.analysis_results('unitary_linear_fit_params').value
    slope, intercept, psi, phi = np.stack([fit_params[ic] for ic in range(3)], axis=1)

    cr_width = runner.calibrations.get_parameter_value('width', qubits, 'cr')
    unitary_axes = np.stack([unp.sin(psi) * unp.cos(phi), unp.sin(psi) * unp.sin(phi), unp.cos(psi)],
                            axis=-1)
    cr_params = unp.nominal_values((slope * cr_width + intercept)[..., None] * unitary_axes)

    omega_z = (np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) @ (slope * unp.cos(psi))
               / runner.backend.dt)

    if setup_sizzle(omega_z, runner):
        config = experiments['c2t_cr_unitaries'](runner)
        config.exp_type += '_postsizzle'
        data = runner.run_experiment(config)
        unitary_params = data.analysis_results('unitary_parameters').value
        cr_params = np.array([unp.nominal_values(unitary_params[ic]) for ic in range(3)])

    rotary_angle = get_rotary_guess(cr_params, runner)
    rotary_amp = rotary_angle / rotary_angle_per_amp(runner.backend, runner.calibrations, qubits)
    sign_angle = 0.
    if rotary_amp < 0.:
        sign_angle = np.pi
        rotary_amp *= -1.
    runner.calibrations.add_parameter_value(rotary_amp, 'counter_amp', qubits, schedule='cr')
    runner.calibrations.add_parameter_value(sign_angle, 'counter_sign_angle', qubits, schedule='cr')


def setup_sizzle(omega_z: np.ndarray, runner: ExperimentsRunner) -> bool:
    qubits = tuple(runner.program_data['qubits'][1:])
    # If |omega_Iz| < 3σ(omega_Iz), no need to try siZZle
    if np.abs(omega_z[0].n) < 3. * omega_z[0].std_dev:
        return False

    # Try siZZle to cancel the Z components
    if (sizzle_params := get_sizzle_params(unp.nominal_values(omega_z), runner)) is None:
        return False

    runner.calibrations.add_parameter_value(sizzle_params['frequency'], 'stark_frequency', qubits,
                                            'cr')
    runner.run_experiment('c2t_sizzle_t_amp_scan')
    counter_stark_amp = runner.calibrations.get_parameter_value('counter_stark_amp', qubits, 'cr')
    if counter_stark_amp == 0.:
        return False

    cr_amp = runner.calibrations.get_parameter_value('cr_amp', qubits, 'cr')
    if abs(sizzle_params['c_amp'] * sizzle_params['t_amp'] / counter_stark_amp) < 1. - cr_amp:
        runner.run_experiment('c2t_sizzle_c2_amp_scan')

    return True


def get_sizzle_params(z_comps, runner):
    hvars = runner.backend.configuration().hamiltonian['vars']
    control2, target = runner.program_data['qubits'][1:]

    def get_shifts(amps, freq, phase=0.):
        return sizzle_hamiltonian_shifts(hvars, (control2, target), amps, freq, phase=phase)

    intervals = frequency_intervals(runner)

    # Coarse grid search
    frequencies = np.concatenate([np.arange(v[0], v[1], 1.e+6) for v in intervals])
    amplitudes = np.linspace(-1., 1., 20)
    all_shifts = np.array(
        [[get_shifts((c, t), frequencies) for t in amplitudes] for c in amplitudes]
    )
    ic, it, ifreq = np.unravel_index(
        np.argmin(np.sum(np.square(all_shifts[..., 1] + z_comps), axis=-1)),
        all_shifts.shape[:-2]
    )
    # Minimization
    def objective(params):
        freq, camp, tamp = np.asarray(params) * np.array([1.e+9, 1., 1.])
        z_shifts = get_shifts((camp, tamp), freq)[..., 1]
        return np.sum(np.square(z_shifts + z_comps), axis=-1)

    freq_interval = next(tuple(np.array(v) * 1.e-9) for v in intervals
                         if v[0] <= frequencies[ifreq] <= v[1])
    res = sciopt.minimize(objective, (frequencies[ifreq] * 1.e-9, amplitudes[ic], amplitudes[it]),
                          bounds=[freq_interval, (-1., 1.), (-1., 1.)])

    return {
        'frequency': res.x[0] * 1.e+9,
        'c_amp': res.x[1],
        't_amp': res.x[2]
    }


def frequency_intervals(runner):
    """Return disjoint frequency intervals around GE and EF resonances of qubits connected to c2."""
    control2 = runner.program_data['qubits'][1]
    neighbors = set(
        sum((edge for edge in runner.backend.coupling_map.get_edges() if control2 in edge), ())
    )
    qubit_props = [runner.backend.qubit_properties(q) for q in neighbors]
    qubit_frequencies = np.array([p.frequency for p in qubit_props])
    anharmonicities = np.array([p.anharmonicity for p in qubit_props])
    resonances = np.sort(np.concatenate([qubit_frequencies, qubit_frequencies + anharmonicities]))
    intervals = [(resonances[0] - 50.e+6, resonances[0] - 15.e+6)]
    intervals += [(left + 15.e+6, right - 15.e+6)
                  for left, right in zip(resonances[:-1], resonances[1:])]
    intervals += [(resonances[-1] + 15.e+6, resonances[-1] + 50.e+6)]
    return intervals


def get_rotary_guess(
    cr_params: np.ndarray,
    runner: ExperimentsRunner,
    scan_range: tuple[float, float] = (-6. * np.pi, 6. * np.pi)
):
    """Find the rotary angle for the current CR parameters that is expected to minimize y and z on
    CRCR."""
    qubits = tuple(runner.program_data['qubits'][1:])
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', qubits)

    rotary_angles = np.linspace(scan_range[0], scan_range[1], 400)
    rotary = np.stack([rotary_angles] + [np.zeros_like(rotary_angles)] * 2, axis=-1)
    cr_params = cr_params[:, None, :] + rotary[None, ...]
    cr_params = np.stack([cr_params, cr_params * np.array([-1., -1., 1.])], axis=1)
    cr = so3_cartesian(cr_params) # [control, sign, rotary, 3, 3]
    if rcr_type == RCRType.X:
        rcr = np.array([
            cr[1, :, :] @ cr[0, :, :],
            cr[0, :, :] @ cr[1, :, :],
            cr[2, :, :] @ cr[2, :, :]
        ])
        sgn = [0, 1, 0]
    else:
        rcr = np.array([
            cr[0, :, :] @ cr[0, :, :],
            cr[2, :, :] @ cr[1, :, :],
            cr[1, :, :] @ cr[2, :, :]
        ])
        sgn = [0, 0, 1]

    crcr = np.array([
        rcr[(ic + 2) % 3, sgn[2]] @ rcr[(ic + 1) % 3, sgn[1]] @ rcr[ic, sgn[0]]
        for ic in range(3)
    ])
    crcr_params = so3_cartesian_params(crcr) # [control, rotary, 3]
    irotary = np.argmin(np.sum(np.square(crcr_params[..., 1:]), axis=(0, 2)))
    return rotary_angles[irotary]
