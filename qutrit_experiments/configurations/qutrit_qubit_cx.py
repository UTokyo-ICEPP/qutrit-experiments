# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
"""Experiment configurations for qutrit-qubit CX gate calibration."""
from functools import wraps
import logging
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit

from ..calibrations import get_qutrit_pulse_gate, get_qutrit_qubit_composite_gate
from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..gates import CRCRGate, CrossResonanceGate, GateType, QutritGate, QutritQubitCXGate, RCRGate
from ..util.pulse_area import gs_effective_duration, rabi_cycles_per_area
from .common import add_readout_mitigation, qubits_assignment_error, qubits_assignment_error_post

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


@register_exp
@wraps(qubits_assignment_error)
def qubits_assignment_error_func(runner):
    return qubits_assignment_error(runner, runner.qubits)

register_post(qubits_assignment_error_post, exp_type='qubits_assignment_error')

def unitaries(runner, gate):
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    if isinstance(gate, QutritGate):
        if gate.gate_type == GateType.COMPOSITE:
            sched = get_qutrit_qubit_composite_gate(gate.name, runner.qubits, runner.calibrations,
                                                    target=runner.backend.target)
        elif gate.gate_type == GateType.PULSE:
            qutrit = np.array(runner.qubits)[list(gate.as_qutrit)][0]
            sched = get_qutrit_pulse_gate(gate.name, qutrit, runner.calibrations,
                                          target=runner.backend.target)
    else:
        sched = runner.calibrations.get_schedule(gate.name, runner.qubits)

    circuit = QuantumCircuit(2)
    circuit.append(gate, [0, 1])
    circuit.add_calibration(gate.name, runner.qubits, sched)

    return ExperimentConfig(
        QutritQubitTomography,
        runner.qubits,
        args={'circuit': circuit, 'measure_qutrit': True},
        run_options={'shots': 8000}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cr_unitaries(runner):
    return unitaries(runner, CrossResonanceGate())

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def rcr_unitaries(runner):
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    return unitaries(runner, RCRGate.of_type(rcr_type)())

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def crcr_unitaries(runner):
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    return unitaries(runner, CRCRGate.of_type(rcr_type)())

@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cx_unitaries(runner):
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    return unitaries(runner, QutritQubitCXGate.of_type(rcr_type)())

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
            'widths': np.arange(128, 384, 64)
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
    def cal_criterion(data):
        return data.analysis_results('pi_amp', block=False).value.n < 0.95

    return ExperimentConfig(
        CRRoughAmplitudeCal,
        runner.qubits,
        calibration_criterion=cal_criterion
    )

@register_post
def rcr_rough_cr_amp(runner, experiment_data):
    fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
    rcr_type = runner.calibrations.get_parameter_value('rcr_type', runner.qubits)
    runner.program_data['crcr_dxda'] = np.array(
        [2. * fit_params[rcr_type][0].n, 2. * fit_params[1][0].n - fit_params[rcr_type][0].n]
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def rcr_rotary_amp(runner):
    """Rotary tone amplitude calibration to minimize the y and z components of RCR."""
    from ..experiments.qutrit_qubit_cx.rotary import RepeatedCRRotaryAmplitudeCal
    duration = gs_effective_duration(runner.calibrations, runner.qubits, 'cr')
    cycles_per_amp = rabi_cycles_per_area(runner.backend, runner.qubits[1]) * duration
    cycles_per_amp *= 2 # Factor two because RCR = CR * 2
    amplitudes = np.linspace(0.5, 2.5, 20) / cycles_per_amp
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.qubits,
        args={'amplitudes': amplitudes},
        analysis_options={'parallelize': -1, 'parallelize_on_thread': True}
    )

@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_scanbased(runner):
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineScanCal
    return ExperimentConfig(
        CycledRepeatedCRFineScanCal,
        runner.qubits
    )
