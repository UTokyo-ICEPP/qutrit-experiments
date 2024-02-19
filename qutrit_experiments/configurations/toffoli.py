# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for Toffoli gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp

from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..experiments.qutrit_qubit_cx.util import RCRType, make_crcr_circuit
from ..util.pulse_area import rabi_freq_per_amp, grounded_gauss_area
from .common import add_readout_mitigation, qubits_assignment_error
from .qutrit import (
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
)

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


def register_single_qutrit_exp(function):
    @wraps(function)
    def conf_gen(runner):
        return function(runner, runner.program_data['qubits'][1])

    register_exp(conf_gen)

qutrit_functions = [
    qutrit_rough_frequency,
    qutrit_rough_amplitude,
    qutrit_semifine_frequency,
    qutrit_fine_frequency,
    qutrit_rough_x_drag,
    qutrit_rough_sx_drag,
    qutrit_fine_sx_amplitude,
    qutrit_fine_sx_drag,
    qutrit_fine_x_amplitude,
    qutrit_fine_x_drag,
    qutrit_x12_stark_shift,
    qutrit_x_stark_shift,
    qutrit_sx_stark_shift,
    qutrit_rotary_stark_shift
]
for func in qutrit_functions:
    register_single_qutrit_exp(func)


@register_exp
@wraps(qubits_assignment_error)
def qubits_assignment_error_func(runner):
    return qubits_assignment_error(runner, runner.program_data['qubits'])

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_cr_rough_width(runner):
    """Few-sample CR UT to measure ωx to find a rough estimate for the CR width in CRCR."""
    from ..experiments.qutrit_qubit_cx.cr_width import CRRoughWidthCal
    return ExperimentConfig(
        CRRoughWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'widths': np.arange(128., 384., 64.)
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_sizzle_t_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRTargetStarkCal
    return ExperimentConfig(
        QutritCRTargetStarkCal,
        runner.program_data['qubits'][1:]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_sizzle_c2_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRControlStarkCal
    return ExperimentConfig(
        QutritCRControlStarkCal,
        runner.program_data['qubits'][1:]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_cr_width(runner):
    from ..experiments.qutrit_qubit_cx.cr_width import CycledRepeatedCRWidthCal
    qubits = runner.program_data['qubits'][1:]

    current_width = runner.calibrations.get_parameter_value('width', qubits, schedule='cr')
    if current_width != 0.:
        widths = np.linspace(current_width - 128, current_width + 128, 5)
        while widths[0] < 0.:
            widths += 16.
    else:
        widths = None

    return ExperimentConfig(
        CycledRepeatedCRWidthCal,
        runner.program_data['qubits'][1:],
        args={
            'widths': widths,
        }
    )

@register_post
def c2t_crcr_cr_width(runner, experiment_data):
    # d|θ_1x - θ_0x|/dt
    fit_params = experiment_data.analysis_results('unitary_linear_fit_params', block=False).value
    params = [unp.nominal_values(fit_params[ic]) for ic in range(2)]
    runner.program_data['crcr_width_rate_params'] = np.array(
        [np.array([p[0], p[1]]) * np.sin(p[2]) * np.cos(p[3]) for p in params]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_rcr_rotary(runner):
    """Fine-tuning the rotary amplitude to squeeze out the y component in RCR. Unused."""
    from ..experiments.qutrit_qubit_cx.rotary import RepeatedCRRotaryAmplitudeCal
    qubits = runner.program_data['qubits'][1:]

    sigma = runner.calibrations.get_parameter_value('sigma', qubits, 'cr')
    rsr = runner.calibrations.get_parameter_value('rsr', qubits, 'cr')
    width = runner.calibrations.get_parameter_value('width', qubits, 'cr')
    gs_area = grounded_gauss_area(sigma, rsr, gs_factor=True) + width
    angle_per_amp = (rabi_freq_per_amp(runner.backend, qubits[1]) * twopi * runner.backend.dt
                     * gs_area)
    angle_per_amp *= 2. # Un-understood empirical factor 2
    if (angles := runner.program_data.get('rcr_rotary_test_angles')) is None:
        # Scan rotary amplitudes expected to generate +-1 rad rotations within one CR pulse
        angles = np.linspace(-1., 1., 8)

    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:],
        args={'amplitudes': angles / angle_per_amp},
        analysis_options={'thetax_per_amp': angle_per_amp}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_rotary(runner):
    from ..experiments.qutrit_qubit_cx.rotary import CycledRepeatedCRRotaryAmplitudeCal
    qubits = runner.program_data['qubits'][1:]

    sigma = runner.calibrations.get_parameter_value('sigma', qubits, 'cr')
    rsr = runner.calibrations.get_parameter_value('rsr', qubits, 'cr')
    width = runner.calibrations.get_parameter_value('width', qubits, 'cr')
    gs_area = grounded_gauss_area(sigma, rsr, gs_factor=True) + width
    angle_per_amp = (rabi_freq_per_amp(runner.backend, qubits[1]) * twopi * runner.backend.dt
                     * gs_area)
    angle_per_amp *= 2. # Un-understood empirical factor 2
    if (angles := runner.program_data.get('crcr_rotary_test_angles')) is None:
        # Scan rotary amplitudes expected to generate +-1 rad rotations within one CR pulse
        angles = np.linspace(-1., 1., 8)

    return ExperimentConfig(
        CycledRepeatedCRRotaryAmplitudeCal,
        runner.program_data['qubits'][1:],
        args={'amplitudes': angles / angle_per_amp}
    )

@register_post
def c2t_crcr_rotary(runner, experiment_data):
    qubits = tuple(runner.program_data['qubits'][1:])
    rotary_amp = runner.calibrations.get_parameter_value('counter_amp', qubits, 'cr')
    if runner.calibrations.get_parameter_value('counter_sign_angle', qubits, 'cr') != 0.:
        rotary_amp *= -1.
    rotary_idx = int(np.argmin(np.abs(runner.program_data['crcr_rotary_test_angles'] - rotary_amp)))
    child_0 = experiment_data.child_data(rotary_idx).child_data(0)
    fit_params = child_0.analysis_results('unitary_fit_params').value
    runner.program_data['rx_target_angle'] = -fit_params[0].n

@register_exp
@add_readout_mitigation
def c2t_crcr_rx_amp(runner):
    from ..experiments.qutrit_qubit_cx.rx_amp import SimpleRxAmplitudeCal
    qubits = (runner.program_data['qubits'][2],)
    x_sched = runner.backend.defaults().instruction_schedule_map.get('x', qubits)
    pi_amp = x_sched.instructions[0][1].pulse.amp

    return ExperimentConfig(
        SimpleRxAmplitudeCal,
        qubits,
        args={
            'target_angle': runner.program_data['rx_target_angle'],
            'amplitudes': np.linspace(-pi_amp, pi_amp, 32)
        }
    )

@register_post
def c2t_crcr_rx_amp(runner, experiment_data):
    angular_rate = experiment_data.analysis_results('rabi_rate', block=False).value.n * twopi
    runner.program_data['crcr_angle_per_rx_amp'] = angular_rate

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def c2t_crcr_fine(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineCal
    return ExperimentConfig(
        CycledRepeatedCRFineCal,
        runner.program_data['qubits'][1:],
        args={
            'width_rate_params': runner.program_data['crcr_width_rate_params'],
            'amp_rate': runner.program_data['crcr_angle_per_rx_amp']
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def c2t_crcr_validation(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    qubits = tuple(runner.program_data['qubits'][1:])
    cr_schedules = [runner.calibrations.get_schedule('cr', qubits)]
    # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
    assign_params = {pname: np.pi for pname in
                    ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']}
    cr_schedules.append(runner.calibrations.get_schedule('cr', qubits, assign_params=assign_params))
    rx_schedule = runner.calibrations.get_schedule('offset_rx', qubits[1])
    rcr_type = RCRType(runner.calibrations.get_parameter_value('rcr_type', qubits))
    crcr_circuit = make_crcr_circuit(qubits, cr_schedules, rx_schedule, rcr_type)

    return ExperimentConfig(
        QutritQubitTomography,
        qubits,
        args={'circuit': crcr_circuit}
    )
