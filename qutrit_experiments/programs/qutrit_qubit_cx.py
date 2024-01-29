import logging
from typing import Any
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX

import qutrit_experiments.configurations.single_qutrit
import qutrit_experiments.configurations.toffoli
from qutrit_experiments.experiment_config import experiments, postexperiments
from qutrit_experiments.experiments.cr_rabi import cr_rabi_init
from qutrit_experiments.runners import ExperimentsRunner
from qutrit_experiments.util.sizzle import sizzle_hamiltonian_shifts

logger = logging.getLogger(__name__)


def calibrate_qutrit_qubit_cx(
    runner: ExperimentsRunner
):
    exp_data = runner.program_data.setdefault('experiment_data', {})

    exp_data['qubits_assignment_error'] = runner.run_experiment('qubits_assignment_error')

    sizzle_params = calibrate_cr(runner)

    if sizzle_params:
        exp_data['c2t_hcr_validation'] = runner.run_experiment('c2t_hcr_validation')
    else:
        postexperiments['c2t_hcr_validation'](runner, exp_data[hcr_exp_type(cr_amp_val)])

    for exp_type in ['c2t_rcr_rotary', 'c2t_crcr_cr_width', 'c2t_crcr_rx_amp']:
        exp_data[exp_type] = runner.run_experiment(exp_type)


def calibrate_cr(runner: ExperimentsRunner, z_comp_threshold: float = 3.5e+5):
    cr_amp_val = 0.9
    while True:
        sizzle_params = try_cr_calibration(cr_amp_val, runner, cr_amp_val > 0.65, z_comp_threshold)
        if sizzle_params is not None:
            break
        cr_amp_val -= 0.1
    else:
        raise RuntimeError('Failed to find a working point for the CR tone.')

    qubits = runner.program_data['qubits'][1:]
    runner.calibrations.add_parameter_value(cr_amp_val, 'cr_amp', qubits=qubits, schedule='cr')
    if sizzle_params: # could be empty
        if sizzle_params['c_amp'] < 0.:
            sizzle_params['c_amp'] *= -1.
            control_sizzle_phase = np.pi
        else:
            control_sizzle_phase = 0.

        update_params = [
            ('stark_frequency', sizzle_params['frequency']),
            ('cr_stark_amp', sizzle_params['c_amp']),
            ('cr_stark_sign_phase', control_sizzle_phase),
            ('counter_stark_amp', sizzle_params['t_amp'])
        ]
    else:
        update_params = [
            ('cr_stark_amp', 0.),
            ('counter_stark_amp', 0.)
        ]
    for pname, value in update_params:
        runner.calibrations.add_parameter_value(value, pname, qubits=qubits, schedule='cr')

    return sizzle_params


def try_cr_calibration(
    cr_amp_val: float,
    runner: ExperimentsRunner,
    large_cr_amp: bool = True,
    z_comp_threshold: float = 3.5e+5
):
    exp_data = runner.program_data.setdefault('experiment_data', {})
    measure_with_offset = measure_qutrit_qubit_hcr(cr_amp_val, runner)

    if measure_with_offset:
        if 't_hx' not in exp_data:
            logger.info('Measuring the offset Rx tone Hamiltonian.')
            exp_data['t_hx'] = runner.run_experiment('t_hx', block_for_results=False)

        logger.info('Measuring CR Hamiltonian with offset for control states %s',
                    measure_with_offset)
        measure_singlestate_hcrs(measure_with_offset, cr_amp_val, runner)

    for data in exp_data.values():
        data.block_for_results()

    z_comps = get_z_comps(cr_amp_val, runner)

    if np.all(np.abs(z_comps) < z_comp_threshold):
        logger.info('All Z components are under threshold without using siZZle.')
        return {}

    sizzle_params = get_sizzle_params(z_comps, runner)
    logger.info('Theoretical prediction of sizzle amplitudes: (%.3f, %.3f)',
                sizzle_params['c_amp'], sizzle_params['t_amp'])

    if (sizzle_params is None
        or (np.any(np.abs(z_comps[1:]) > z_comp_threshold)
            and np.abs(sizzle_params['c_amp']) + cr_amp_val > 2.)):
        logger.info('CR amplitude of %.1f was not viable with available siZZle frequencies',
                    cr_amp_val)
        return None

    z_comps_predicted_shift = np.zeros(3)

    logger.info('Finding the target siZZle amplitude to cancel Iz=%.3e.', z_comps[0])
    iz_shift = measure_iz_shift(sizzle_params, z_comps[0], runner)
    if iz_shift is None:
        logger.warning('Guessed a wrong frequency; Iz dependency wrong sign')
        return None
    z_comps_predicted_shift[0] = iz_shift
    logger.info('Identified target Stark amp of %f generating iz shift of %f',
                sizzle_params['t_amp'], z_comps_predicted_shift[0])

    if np.any(np.abs(z_comps[1:]) > z_comp_threshold):
        logger.info('Finding the control siZZle amplitude to cancel [zz ζz]=%s.', z_comps[1:])
        z_comps_predicted_shift[1:] = measure_zz_zetaz_shifts(sizzle_params, z_comps[1:], runner)
        logger.info('Identified control Stark amp of %f generating z shifts of %s',
                    sizzle_params['c_amp'], z_comps_predicted_shift[1:])

        if np.abs(sizzle_params['c_amp']) + cr_amp_val > 1.:
            logger.info('ZZ values could not be canceled within total amplitude of 1.')
            if large_cr_amp:
                # We should reduce the amplitude and try again
                return None
            else:
                sizzle_params['c_amp'] = (0.99 - cr_amp_val) * np.sign(sizzle_params['c_amp'])
    else:
        sizzle_params['c_amp'] = 0.

    if np.any(np.abs(z_comps + z_comps_predicted_shift) > z_comp_threshold):
        logger.info('Uncancelled Z components remain: %s', z_comps + z_comps_predicted_shift)
        if large_cr_amp:
            return None

    logger.info('Returning siZZle parameters %s.', sizzle_params)
    return sizzle_params


def make_amp_tag(cr_amp_val):
    return f'amp{int(np.round(cr_amp_val * 10)):02d}'

def hcr_exp_type(cr_amp_val):
    return f'c2t_hcr_{make_amp_tag(cr_amp_val)}'

def single_hcr_exp_type(cr_amp_val, control_state):
    return f'c2t_hcr_{make_amp_tag(cr_amp_val)}_control{control_state}_rxoffset'

def target_sizzle_ampscan_exp_type(param_idx):
    return f'c2t_sizzle_t_amplitude_scan_param_{param_idx}'

def control_sizzle_ampscan_exp_type(param_idx):
    return f'c2t_sizzle_c2_amplitude_scan_param_{param_idx}'

def make_hcr_config(cr_amp_val, runner):
    config = experiments['c2t_hcr_template'](runner)
    schedule = config.args['schedule']
    cr_amp = schedule.get_parameters('cr_amp')[0]
    schedule.assign_parameters({cr_amp: cr_amp_val}, inplace=True)
    config.exp_type = hcr_exp_type(cr_amp_val)
    return config

def measure_qutrit_qubit_hcr(cr_amp_val, runner) -> list[int]:
    exp_data = runner.program_data['experiment_data']
    config = make_hcr_config(cr_amp_val, runner)
    exp_data[config.exp_type] = runner.run_experiment(config)

    measure_with_offset = []
    for control_state, child_data in enumerate(exp_data[config.exp_type].child_data()):
        res_name = f'{PARAMS_ENTRY_PREFIX}HamiltonianTomographyAnalysis'
        popt = child_data.analysis_results(res_name).value.params
        if popt['omega'] < 0.5 * np.pi / (2048 * runner.backend.dt):
            measure_with_offset.append(control_state)
    return measure_with_offset

def make_single_hcr_config(cr_amp_val, control_state, runner):
    config = experiments['c2t_hcr_singlestate_template'](runner)
    schedule = config.args['schedule']
    cr_amp = schedule.get_parameters('cr_amp')[0]
    schedule.assign_parameters({cr_amp: cr_amp_val}, inplace=True)
    config.args['rabi_init'] = cr_rabi_init(control_state)
    config.exp_type = single_hcr_exp_type(cr_amp_val, control_state)
    return config

def measure_singlestate_hcrs(measure_with_offset, cr_amp_val, runner):
    for control_state in measure_with_offset:
        config = make_single_hcr_config(cr_amp_val, control_state, runner)
        runner.program_data['experiment_data'][config.exp_type] = \
            runner.run_experiment(config, block_for_results=False)

def get_z_comps(cr_amp_val, runner):
    exp_data = runner.program_data['experiment_data']
    hcr_data = exp_data[hcr_exp_type(cr_amp_val)]
    component_index = hcr_data.metadata["component_child_index"]

    logger.info('Extracting the Z components of the qutrit-qubit CR Hamiltonian')

    control_basis_components = np.empty((3, 3), dtype=object)
    for control_state in range(3):
        data = hcr_data.child_data(component_index[control_state])
        components = np.array(data.analysis_results('hamiltonian_components').value)
        if (data := exp_data.get(single_hcr_exp_type(cr_amp_val, control_state))) is not None:
            components_alt = np.array(data.analysis_results('hamiltonian_components').value)
            components_alt -= unp.nominal_values(runner.program_data['target_rxtone_hamiltonian'])
            logger.info('Control state %d: replacing original components\n%s\nwith\n%s',
                        control_state, components, components_alt)
            components = components_alt

        # omega^c_g/2
        control_basis_components[control_state] = components

    control_eigvals = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) # [c, I/z/ζ]
    # Multiply by factor two to obtain omega_[Izζ]
    hamiltonian = (np.linalg.inv(control_eigvals) @ control_basis_components) * 2
    logger.info('Hamiltonian components:\n%s', hamiltonian)
    return unp.nominal_values(hamiltonian[:, 2])

def get_sizzle_params(z_comps, runner):
    hvars = runner.backend.configuration().hamiltonian['vars']
    control2, target = runner.program_data['qubits'][1:]

    def get_shifts(amps, freq, phase=0.):
        return sizzle_hamiltonian_shifts(hvars, (control2, target), amps, freq, phase=phase)

    test_freqs = sizzle_frequency_candidates(runner)
    shifts = get_shifts((0., 0.02), test_freqs)

    def objective(amps, frequency, phase):
        z_shifts = get_shifts(amps, frequency, phase)[..., 1]
        return np.sum(np.square(z_shifts + z_comps), axis=-1)

    iz_viable = np.nonzero(shifts[:, 0, 1] * z_comps[0] < 0.)[0]

    parameters = None
    minfval = -1.
    for itest in iz_viable:
        frequency = test_freqs[itest]
        for isign, sign_phase in enumerate([0., np.pi]):
            res = sciopt.minimize(objective, (0.2, 0.05), args=(frequency, sign_phase),
                                  bounds=[(0., 1.), (0., 1.)])
            if res.success and (minfval < 0. or res.fun < minfval):
                parameters = (itest * 2 + isign, frequency, res.x[0], res.x[1])
                minfval = res.fun

    if minfval < 0.:
        return None

    return {
        'param_idx': parameters[0],
        'frequency': parameters[1],
        'c_amp': parameters[2] * (1. if parameters[0] % 2 == 0. else -1.),
        't_amp': parameters[3]
    }

def sizzle_frequency_candidates(runner):
    control2, target = runner.program_data['qubits'][1:]
    c2_props = runner.backend.qubit_properties(control2)
    t_props = runner.backend.qubit_properties(target)
    resonances = np.array([c2_props.frequency, t_props.frequency])
    anharmonicities = np.array([c2_props.anharmonicity, t_props.anharmonicity])
    resonances = np.concatenate([resonances, resonances + anharmonicities])
    return np.sort(np.concatenate([resonances - 15.e+6, resonances + 15.e+6]))

def make_target_sizzle_config(sizzle_params, runner):
    target_sizzle_amps = np.linspace(0.01, 0.16, 6)
    config = experiments['c2t_sizzle_amplitude_scan_template'](runner)
    config.args['frequency'] = sizzle_params['frequency']
    config.args['amplitudes'] = (0., target_sizzle_amps)
    config.args['measure_shift'] = False
    config.exp_type = target_sizzle_ampscan_exp_type(sizzle_params['param_idx'])
    return config

def measure_iz_shift(sizzle_params, cr_iz, runner):
    exp_data = runner.program_data['experiment_data']
    if (data := exp_data.get(target_sizzle_ampscan_exp_type(sizzle_params['param_idx']))) is None:
        config = make_target_sizzle_config(sizzle_params, runner)
        data = exp_data[config.exp_type] = runner.run_experiment(config)

    popt = data.analysis_results('ω_Iz_coeffs').value
    # We know that Iz must be zero for zero amp, so essentially treat c as the static offset
    if (asq := -cr_iz / popt['a'].n) < 0.:
        return None
    sizzle_params['t_amp'] = np.sqrt(asq)
    return popt['a'].n * sizzle_params['t_amp'] ** 2

def make_control_sizzle_config(sizzle_params, runner):
    control_sizzle_amps = np.linspace(0.01, 0.41, 5)
    config = experiments['c2t_sizzle_amplitude_scan_template'](runner)
    config.args['frequency'] = sizzle_params['frequency']
    config.args['amplitudes'] = (control_sizzle_amps, sizzle_params['t_amp'])
    base_angles = config.args['angles']
    config.args['angles'] = (base_angles[0] + (1. - np.sign(sizzle_params['c_amp'])) * 0.5 * np.pi,
                             base_angles[1])
    config.args['measure_shift'] = False
    config.exp_type = control_sizzle_ampscan_exp_type(sizzle_params['param_idx'])
    return config

def measure_zz_zetaz_shifts(sizzle_params, cr_zs, runner):
    exp_data = runner.program_data['experiment_data']
    if (data := exp_data.get(control_sizzle_ampscan_exp_type(sizzle_params['param_idx']))) is None:
        config = make_control_sizzle_config(sizzle_params, runner)
        data = exp_data[config.exp_type] = runner.run_experiment(config)
    if 'c2t_static_omega_zs' not in runner.program_data:
        exp_data['c2t_zzramsey'] = runner.run_experiment('c2t_zzramsey')

    fit_popts = [data.analysis_results(f'ω_{op}_coeffs').value for op in ['zz', 'ζz']]
    slopes = np.array([popt['slope'].n for popt in fit_popts])
    intercepts = np.array([popt['intercept'].n for popt in fit_popts])
    # We know that ζz is zero for zero amp so subtract the ζz as the static offset
    intercepts[0] -= runner.program_data['c2t_static_omega_zs'][1].n
    intercepts[1] = 0.
    # min sum_{i}(z_i + a_i*x + b_i)^2
    # -> 2*sum_{i}(a_i*(z_i + a_i*x + b_i)) = 0
    # -> x = -[sum_{i}(a_i*(z_i + b_i))] / [sum_{i}a_i^2]
    sizzle_params['c_amp'] = -np.sum(slopes * (cr_zs + intercepts)) / np.sum(np.square(slopes))
    return slopes * sizzle_params['c_amp'] + intercepts
