import logging
from typing import Any
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.framework import BackendTiming

from ..configurations.toffoli import c2t_sizzle_c2_amp_scan_template, c2t_sizzle_t_amp_scan_template
from ..experiment_config import BatchExperimentConfig, ExperimentConfig, experiments, postexperiments
from ..experiments.cr_rabi import cr_rabi_init
from ..experiments.unitary_tomography import UnitaryTomography
from ..runners import ExperimentsRunner
from ..util.pulse_area import rabi_freq_per_amp, grounded_gauss_area
from ..util.sizzle import sizzle_hamiltonian_shifts

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner,
    initial_cr_amp: float = 0.8
):
    exp_data = runner.program_data.setdefault('experiment_data', {})
    qubits = runner.program_data['qubits'][1:]
    runner.program_data['initial_cr_amp'] = initial_cr_amp

    # Construct the error mitigation matrix and find the rough CR pulse width
    for exp_type in ['qubits_assignment_error', 'c2t_cr_rough_width']:
        exp_data[exp_type] = runner.run_experiment(exp_type)

    # Check the Z components of the CR Hamiltonian and decide whether to tune down the amplitude or
    # introduce siZZle
    cr_params = {
        'width': runner.program_data['crcr_pi_width'],
        'margin': 0.,
        'stark_frequency': runner.backend.qubit_properties(qubits[1]).frequency,
        'cr_amp': initial_cr_amp,
        'cr_sign_angle': 0.,
        'cr_stark_amp': 0.,
        'cr_stark_sign_phase': 0.,
        'counter_amp': 0.,
        'counter_stark_amp': 0.
    }
    fine_tune_cr(runner, cr_params)

    # Inject the calibration parameters
    for pname, value in cr_params.items():
        runner.calibrations.add_parameter_value(value, pname, qubits, 'cr')

    for exp_type in ['c2t_rcr_rotary', 'c2t_crcr_cr_width', 'c2t_crcr_rx_amp']:
        exp_data[exp_type] = runner.run_experiment(exp_type)

def fine_tune_cr(runner: ExperimentsRunner, cr_params: dict[str, float]):
    exp_data = runner.program_data['experiment_data']
    cr_omegas = exp_data['c2t_cr_rough_width'].analysis_results('hamiltonian_components').value
    cr_omegas = unp.nominal_values(cr_omegas)

    theta_z = cr_omegas[:, 2] * (cr_params['width'] + gauss_flank_area(runner)) * runner.backend.dt
    if np.all(np.abs(theta_z) < 0.05):
        return

    # Otherwise first check the induced Zs
    exp_data['c2t_zzramsey'] = runner.run_experiment('c2t_zzramsey')
    static_omega_zs = unp.nominal_values(runner.program_data['c2t_static_omega_zs'])
    induced_omega_z = cr_omegas[:, 2] - static_omega_zs
    flank = gauss_flank_area(runner)
    induced_theta_z = induced_omega_z * (cr_params['width'] + flank) * runner.backend.dt

    if (max_induced := np.max(np.abs(induced_theta_z))) > 0.05:
        scale_down_cr_amp(runner, cr_params, cr_omegas, static_omega_zs, 0.05 / max_induced)
        theta_z = cr_omegas[:, 2] * (cr_params['width'] + flank) * runner.backend.dt
        if np.all(np.abs(theta_z) < 0.05):
            return

    # Try siZZle to cancel the Z components
    sizzle_params = get_sizzle_params(cr_omegas[:, 2], runner)
    cr_params['stark_frequency'] = sizzle_params['frequency']

    # Determine the target drive amplitude that gives the minimum |Iz|
    sizzle_t_amp_scan(runner, cr_params)
    # Determine the control drive amplitude that gives the minimum sqrt(zz^2+ζz^2)
    sizzle_c2_amp_scan(runner, cr_params)

def gauss_flank_area(runner):
    qubits = tuple(runner.program_data['qubits'][1:])
    sigma = runner.calibrations.get_parameter_value('sigma', qubits, 'cr')
    rsr = runner.calibrations.get_parameter_value('rsr', qubits, 'cr')
    return grounded_gauss_area(sigma, rsr, True)

def scale_down_cr_amp(
    runner: ExperimentsRunner,
    cr_params: dict[str, float],
    cr_omegas: np.ndarray,
    static_omega_zs: np.ndarray,
    omega_z_scale: float
):
    # Reduce the cr amplitude
    # Induced ωz should have form a*cr_amp^2
    amp_scale = np.sqrt(omega_z_scale)
    cr_params['cr_amp'] *= amp_scale
    cr_omegas[:, :2] *= amp_scale
    cr_omegas[:, 2] = (cr_omegas[:, 2] - static_omega_zs) * omega_z_scale + static_omega_zs
    flank = gauss_flank_area(runner)
    new_width = (cr_params['width'] + flank) / amp_scale - flank
    cr_params['width'] = BackendTiming(runner.backend).round_pulse(samples=new_width)

def sizzle_t_amp_scan(runner: ExperimentsRunner, cr_params: dict[str, float]):
    exp_data = runner.program_data['experiment_data']
    config = c2t_sizzle_t_amp_scan_template(runner)
    config.exp_type = 'c2t_sizzle_t_amp_scan'
    schedule = list(config.args['circuit'].calibrations['cr'].values())[0]
    assignment = {p: cr_params[p.name] for p in schedule.parameters
                  if p.name != 'counter_stark_amp'}
    schedule.assign_parameters(assignment, inplace=True)
    config.args['values'] = np.linspace(0.005, 0.16, 10)
    exp_data[config.exp_type] = runner.run_experiment(config)

    unitaries = exp_data[config.exp_type].analysis_results('unitary_parameters').value
    theta_iz = np.einsum('oc,sck->sok',
                         np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                         unitaries)[:, 0, 2]
    cr_params['counter_stark_amp'] = config.args['values'][np.argmin(np.abs(theta_iz))]

def sizzle_c2_amp_scan(runner, cr_params):
    exp_data = runner.program_data['experiment_data']
    config = c2t_sizzle_c2_amp_scan_template(runner)
    config.exp_type = 'c2t_sizzle_c2_amp_scan'
    schedule = list(config.args['circuit'].calibrations['cr'].values())[0]
    assignment = {p: cr_params[p.name] for p in schedule.parameters
                  if p.name not in ['cr_stark_amp', 'cr_stark_sign_phase']}
    schedule.assign_parameters(assignment, inplace=True)
    config.args['values'] = np.linspace(
        max(cr_params['cr_amp'] - 0.99, -0.4),
        min(0.99 - cr_params['cr_amp'], 0.4),
        20
    )
    exp_data[config.exp_type] = runner.run_experiment(config)

    unitaries = exp_data[config.exp_type].analysis_results('unitary_parameters').value
    theta_zs = np.einsum('oc,sck->sok',
                         np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                         unitaries)[:, 1:, 2]
    min_idx = np.argmin(np.sum(np.square(theta_zs), axis=-1))
    cr_params['cr_stark_amp'] = config.args['values'][min_idx]

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
    control2, target = runner.program_data['qubits'][1:]
    c2_props = runner.backend.qubit_properties(control2)
    t_props = runner.backend.qubit_properties(target)
    resonances = np.array([c2_props.frequency, t_props.frequency])
    anharmonicities = np.array([c2_props.anharmonicity, t_props.anharmonicity])
    resonances = np.sort(np.concatenate([resonances, resonances + anharmonicities]))
    intervals = [(resonances[0] - 50.e+6, resonances[0] - 15.e+6)]
    intervals += [(left + 15.e+6, right - 15.e+6)
                  for left, right in zip(resonances[:-1], resonances[1:])]
    intervals += [(resonances[-1] + 15.e+6, resonances[-1] + 50.e+6)]
    return intervals
