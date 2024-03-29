# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for qutrit-qubit CX gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit

from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..experiments.qutrit_qubit_cx.util import get_cr_schedules
from ..gates import (CrossResonanceGate, CrossResonanceMinusGate, CrossResonancePlusGate,
                     CXOffsetRxGate, QutritQubitCXGate, RCRGate)
from ..util.pulse_area import gs_effective_duration, rabi_cycles_per_area
from .common import add_readout_mitigation, qubits_assignment_error, qubits_assignment_error_post

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


@register_exp
@wraps(qubits_assignment_error)
def qubits_assignment_error_func(runner):
    return qubits_assignment_error(runner, runner.qubits)

register_post(qubits_assignment_error_post, exp_type='qubits_assignment_error')

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cr_unitaries(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    circuit = QuantumCircuit(2)
    circuit.append(CrossResonanceGate(), [0, 1])
    circuit.add_calibration(CrossResonanceGate.gate_name, runner.qubits,
                            runner.calibrations.get_schedule('cr', runner.qubits))

    return ExperimentConfig(
        QutritQubitTomography,
        runner.qubits,
        args={'circuit': circuit},
        run_options={'shots': 8000}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def rcr_unitaries(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    circuit = RCRGate.of_type(rcr_type).decomposition()
    sched_a = runner.calibrations.get_schedule('cr', runner.qubits)
    sign_angle = runner.calibrations.get_parameter_value('counter_sign_angle', runner.qubits, 'cr')
    sched_b = runner.calibrations.get_schedule('cr', runner.qubits,
                                               assign_params={'counter_sign_angle': sign_angle + np.pi})
    circuit.add_calibration('cra', runner.qubits, sched_a)
    circuit.add_calibration('crb', runner.qubits, sched_b)

    return ExperimentConfig(
        QutritQubitTomography,
        runner.qubits,
        args={'circuit': circuit},
        run_options={'shots': 8000}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def crcr_unitaries(runner):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    cr_schedules = get_cr_schedules(runner.calibrations, runner.qubits)
    rx_schedule = runner.calibrations.get_schedule('cx_offset_rx', runner.qubits[1:])
    circuit = QutritQubitCXGate.of_type(rcr_type).decomposition()
    circuit.add_calibration(CrossResonancePlusGate.gate_name, runner.qubits, cr_schedules[0])
    circuit.add_calibration(CrossResonanceMinusGate.gate_name, runner.qubits, cr_schedules[1])
    circuit.add_calibration('cx_offset_rx', runner.qubits[1:], rx_schedule)

    return ExperimentConfig(
        QutritQubitTomography,
        runner.qubits,
        args={'circuit': circuit},
        run_options={'shots': 8000}
    )

@register_exp
def cr_initial_amp(runner):
    from ..experiments.qutrit_qubit.cr_initial_amp import CRInitialAmplitudeCal
    assignment_matrix = runner.program_data['qutrit_assignment_matrix'][runner.qubits[0]]
    return ExperimentConfig(
        CRInitialAmplitudeCal,
        runner.qubits,
        analysis_options={'assignment_matrix': assignment_matrix}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cr_rough_width(runner):
    """Few-sample CR UT to measure ωx to find a rough estimate for the CR width in CRCR."""
    from ..experiments.qutrit_qubit_cx.cr_width import CRRoughWidthCal
    return ExperimentConfig(
        CRRoughWidthCal,
        runner.qubits,
        args={
            'widths': np.arange(128., 384., 64.)
        }
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def cr_angle(runner):
    """CR angle calibration to eliminate the y component of RCR non-participating state."""
    from ..experiments.qutrit_qubit.cr_angle import FineCRAngleCal
    control_state = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    return ExperimentConfig(
        FineCRAngleCal,
        runner.qubits,
        args={'control_state': control_state}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def cr_counter_stark_amp(runner):
    """CR counter Stark tone amplitude calibration to eliminate the z component of RCR
    non-participating state."""
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRTargetStarkCal
    control_state = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    return ExperimentConfig(
        QutritCRTargetStarkCal,
        runner.qubits,
        args={'control_states': (control_state,)}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def rcr_rough_cr_amp(runner):
    """CR angle calibration to eliminate the y component of RCR non-participating state."""
    from ..experiments.qutrit_qubit_cx.cr_amp import CRRoughAmplitudeCal
    return ExperimentConfig(
        CRRoughAmplitudeCal,
        runner.qubits
    )

@register_post
def rcr_rough_cr_amp(runner, experiment_data):
    fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    runner.program_data['crcr_dxda'] = np.array(
        [2. * fit_params[rcr_type][0].n, 2. * fit_params[1][0].n - fit_params[rcr_type][0].n]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def sizzle_t_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRTargetStarkCal
    return ExperimentConfig(
        QutritCRTargetStarkCal,
        runner.qubits
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def sizzle_c2_amp_scan(runner):
    from ..experiments.qutrit_qubit.qutrit_cr_sizzle import QutritCRControlStarkCal
    return ExperimentConfig(
        QutritCRControlStarkCal,
        runner.qubits
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def rcr_rotary_amp(runner):
    """Rotary tone amplitude calibration to minimize the y and z components of RCR."""
    from ..experiments.qutrit_qubit_cx.rotary import RepeatedCRRotaryAmplitudeCal
    duration = gs_effective_duration(runner.calibrations, runner.qubits, 'cr')
    cycles_per_amp = rabi_cycles_per_area(runner.backend, runner.qubits[1]) * duration
    cycles_per_amp *= 2 # Factor two because RCR = CR * 2
    amplitudes = np.linspace(2.5, 3.5, 20) / cycles_per_amp
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.qubits,
        args={'amplitudes': amplitudes},
        analysis_options={'parallelize': -1, 'parallelize_on_thread': True}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def crcr_rotary(runner):
    from ..experiments.qutrit_qubit_cx.rotary import (CycledRepeatedCRRotaryAmplitudeCal,
                                                      rotary_angle_per_amp)
    angle_per_amp = rotary_angle_per_amp(runner.backend, runner.calibrations, runner.qubits)
    # crcr_rotary_test_angles are the rotary angles with the current CR parameters
    if (angles := runner.program_data.get('crcr_rotary_test_angles')) is None:
        # Scan rotary amplitudes expected to generate +-1 rad rotations within one CR pulse
        angles = np.linspace(-1., 1., 8)

    return ExperimentConfig(
        CycledRepeatedCRRotaryAmplitudeCal,
        runner.qubits,
        args={'amplitudes': angles / angle_per_amp}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_scanbased(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineScanCal
    return ExperimentConfig(
        CycledRepeatedCRFineScanCal,
        runner.qubits
    )

@register_post
def crcr_angle_width_rate(runner, experiment_data):
    # d|θ_1x - θ_0x|/dt
    fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
    slope, _, psi, phi = np.stack([unp.nominal_values(fit_params[ic]) for ic in range(2)], axis=1)
    runner.program_data['crcr_angle_per_width'] = slope * np.sin(psi) * np.cos(phi)

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_iter1(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineCal
    return ExperimentConfig(
        CycledRepeatedCRFineCal,
        runner.qubits,
        args={
            'width_rate': runner.program_data['crcr_angle_per_width'],
            'current_cal_groups': ('crcr_cr_width', 'crcr_rotary')
        },
        run_options={'shots': 8000}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_iter2(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineCal
    return ExperimentConfig(
        CycledRepeatedCRFineCal,
        runner.qubits,
        args={
            'width_rate': runner.program_data['crcr_angle_per_width'],
            'current_cal_groups': ('crcr_fine_iter1', 'crcr_fine_iter1')
        },
        run_options={'shots': 8000}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_iterrx(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineRxAngleCal
    return ExperimentConfig(
        CycledRepeatedCRFineRxAngleCal,
        runner.qubits,
        args={
            'current_cal_group': 'crcr_fine_iter2'
        }
    )
