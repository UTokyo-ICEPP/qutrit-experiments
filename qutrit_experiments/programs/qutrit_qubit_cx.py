import logging
from typing import Union
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit_experiments.framework import BackendTiming

from ..runners import ExperimentsRunner
from ..util.bloch import so3_cartesian_axnorm
from ..util.pulse_area import grounded_gauss_area
from ..util.sizzle import sizzle_hamiltonian_shifts

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner,
    initial_cr_amp: float = 0.8
):
    exp_data = runner.program_data.setdefault('experiment_data', {})
    qubits = runner.program_data['qubits'][1:]

    runner.calibrations.add_parameter_value(initial_cr_amp, 'cr_amp', qubits, 'cr')

    # Construct the error mitigation matrix and find the rough CR pulse width
    for exp_type in ['qubits_assignment_error', 'c2t_cr_rough_width']:
        exp_data[exp_type] = runner.run_experiment(exp_type)

    # Check the Z components of the CR Hamiltonian and decide whether to tune down the amplitude or
    # introduce siZZle
    fine_tune_cr(runner)

    for exp_type in ['c2t_rcr_rotary', 'c2t_crcr_cr_width', 'c2t_crcr_rx_amp']:
        exp_data[exp_type] = runner.run_experiment(exp_type)


def fine_tune_cr(runner: ExperimentsRunner):
    exp_data = runner.program_data['experiment_data']
    qubits = tuple(runner.program_data['qubits'][1:])
    cr_width = runner.calibrations.get_parameter_value('width', qubits, 'cr')

    slope, _, psi, _ = unp.nominal_values(
        exp_data['c2t_cr_rough_width'].analysis_results('unitary_linear_fit_params').value
    ).transpose((1, 0))
    # Disregard the intercept because the logic below is all in terms of omega_z
    # The idea is that by cancelling / suppressing omega_z the intercept should be
    # reduced ~linearly
    theta_z = slope * cr_width * np.cos(psi)
    if np.all(np.abs(theta_z) < 0.05):
        return

    # Otherwise first check the induced Zs
    exp_data['c2t_zzramsey'] = runner.run_experiment('c2t_zzramsey')
    static_omega_z = unp.nominal_values(runner.program_data['c2t_static_omega_zs'])
    omega_z = slope * np.cos(psi) / runner.backend.dt
    induced_omega_z = omega_z - static_omega_z
    flank = gauss_flank_area(runner)
    induced_theta_z = induced_omega_z * (cr_width + flank) * runner.backend.dt

    if (max_induced := np.max(np.abs(induced_theta_z))) > 0.05:
        omega_z = scale_down_cr_amp(runner, omega_z, static_omega_z, 0.05 / max_induced)
        cr_width = runner.calibrations.get_parameter_value('width', qubits, 'cr')
        theta_z = omega_z * (cr_width + flank) * runner.backend.dt

    if np.all(np.abs(theta_z) < 0.05) or np.abs(theta_z[0]) < 0.02:
        # If theta_Iz is too small, target Stark amplitude will be small too, leaving very little
        # room to maneuver with siZZle.
        return

    # Try siZZle to cancel the Z components
    if (sizzle_params := get_sizzle_params(omega_z, runner)) is None:
        return

    runner.calibrations.add_parameter_value(sizzle_params['frequency'], 'stark_frequency', qubits,
                                            'cr')
    exp_data['c2t_sizzle_t_amp_scan'] = runner.run_experiment('c2t_sizzle_t_amp_scan')
    if runner.calibrations.get_parameter_value('counter_stark_amp', qubits, 'cr') > 0.1:
        exp_data['c2t_sizzle_c2_amp_scan'] = runner.run_experiment('c2t_sizzle_c2_amp_scan')


def gauss_flank_area(runner):
    qubits = tuple(runner.program_data['qubits'][1:])
    sigma = runner.calibrations.get_parameter_value('sigma', qubits, 'cr')
    rsr = runner.calibrations.get_parameter_value('rsr', qubits, 'cr')
    return grounded_gauss_area(sigma, rsr, True)


def scale_down_cr_amp(
    runner: ExperimentsRunner,
    omega_z: np.ndarray,
    static_omega_z: np.ndarray,
    omega_z_scale: float
) -> np.ndarray:
    # Reduce the cr amplitude
    # Induced ωz should have form a*cr_amp^2
    amp_scale = np.sqrt(omega_z_scale)
    omega_z = (omega_z - static_omega_z) * omega_z_scale + static_omega_z
    flank = gauss_flank_area(runner)

    qubits = tuple(runner.program_data['qubits'][1:])
    current_amp = runner.calibrations.get_parameter_value('cr_amp', qubits, 'cr')
    runner.calibrations.add_parameter_value(current_amp * amp_scale, 'cr_amp', qubits, 'cr')
    current_width = runner.calibrations.get_parameter_value('width', qubits, 'cr')
    new_width = (current_width + flank) / amp_scale - flank
    new_width = BackendTiming(runner.backend).round_pulse(samples=new_width)
    runner.calibrations.add_parameter_value(new_width, 'width', qubits, 'cr')
    return omega_z


def get_sizzle_params(z_comps, runner):
    hvars = runner.backend.configuration().hamiltonian['vars']
    control2, target = runner.program_data['qubits'][1:]

    def get_shifts(amps, freq, phase=0.):
        return sizzle_hamiltonian_shifts(hvars, (control2, target), amps, freq, phase=phase)

    intervals = frequency_intervals(runner)
    shifts = get_shifts((0., 0.02), np.array([np.mean(interval) for interval in intervals]))
    iz_viable = np.nonzero(shifts[:, 0, 1] * z_comps[0] < 0.)[0]

    def objective(ampsfreq, phase):
        amps = ampsfreq[:2]
        freq = ampsfreq[2] * 1.e+9
        z_shifts = get_shifts(amps, freq, phase)[..., 1]
        return np.sum(np.square(z_shifts + z_comps), axis=-1)

    parameters = None
    minfval = -1.
    for itest in iz_viable:
        freq_interval = np.array(intervals[itest]) * 1.e-9
        for isign, sign_phase in enumerate([0., np.pi]):
            res = sciopt.minimize(objective, (0.2, 0.05, np.mean(freq_interval)), args=(sign_phase,),
                                  bounds=[(0., 1.), (0., 1.), freq_interval])
            if res.success and (minfval < 0. or res.fun < minfval):
                parameters = (itest * 2 + isign, res.x[2] * 1.e+9, res.x[0], res.x[1])
                minfval = res.fun

    if minfval < 0.:
        return None

    return {
        'param_idx': parameters[0],
        'frequency': parameters[1],
        'c_amp': parameters[2] * (1. if parameters[0] % 2 == 0. else -1.),
        't_amp': parameters[3]
    }


def frequency_intervals(runner):
    """Return disjoint frequency intervals around GE and EF resonances of qubits connected to c2."""
    control2 = runner.program_data['qubits'][1]
    edges = [edge for edge in runner.backend.coupling_map.get_edges() if control2 in edge]
    qubit_props = [runner.backend.qubit_properties(control2)]
    for edge in edges:
        if edge[0] == control2:
            qubit_props.append(runner.backend.qubit_properties(edge[1]))
        else:
            qubit_props.append(runner.backend.qubit_properties(edge[0]))

    qubit_frequencies = np.array([p.frequency for p in qubit_props])
    anharmonicities = np.array([p.anharmonicity for p in qubit_props])
    resonances = np.sort(np.concatenate([qubit_frequencies, qubit_frequencies + anharmonicities]))
    intervals = [(resonances[0] - 50.e+6, resonances[0] - 15.e+6)]
    intervals += [(left + 15.e+6, right - 15.e+6)
                  for left, right in zip(resonances[:-1], resonances[1:])]
    intervals += [(resonances[-1] + 15.e+6, resonances[-1] + 50.e+6)]
    return intervals
