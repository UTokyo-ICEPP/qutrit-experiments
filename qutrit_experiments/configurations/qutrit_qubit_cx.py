# pylint: disable=import-outside-toplevel, function-redefined, unused-argument
# pylint: disable=unexpected-keyword-arg, redundant-keyword-arg
"""Experiment configurations for qutrit-qubit CX gate calibration."""

from functools import wraps
import logging
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.data_processing import (BasisExpectationValue, DataProcessor,
                                                MarginalizeCounts, Probability)

from ..calibrations import get_qutrit_pulse_gate, get_qutrit_qubit_composite_gate
from ..experiment_config import ExperimentConfig, register_exp, register_post
from ..gates import CRCRGate, CrossResonanceGate, GateType, QutritGate, QutritQubitCXGate, RCRGate
from ..runners import ExperimentsRunner
from ..util.pulse_area import gs_effective_duration, rabi_cycles_per_area
from .common import add_readout_mitigation, qubits_assignment_error, qubits_assignment_error_post

LOG = logging.getLogger(__name__)
twopi = 2. * np.pi


@register_exp
@wraps(qubits_assignment_error)
def qubits_assignment_error_func(runner: ExperimentsRunner) -> ExperimentConfig:
    return qubits_assignment_error(runner, runner.qubits)


register_post(qubits_assignment_error_post, exp_type='qubits_assignment_error')


def unitaries(
    runner: ExperimentsRunner,
    gate: Gate,
    sched: Optional[ScheduleBlock] = None
) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.qutrit_qubit_tomography import QutritQubitTomography

    if sched is None:
        if isinstance(gate, QutritGate):
            if gate.gate_type == GateType.COMPOSITE:
                sched = get_qutrit_qubit_composite_gate(gate.name, runner.qubits,
                                                        runner.calibrations,
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

    dp_nodes = [MarginalizeCounts({0}), Probability('1'), BasisExpectationValue()]
    assignment_matrix = runner.program_data['qutrit_assignment_matrix'][runner.qubits[0]]

    return ExperimentConfig(
        QutritQubitTomography,
        runner.qubits,
        args={'circuit': circuit, 'measure_qutrit': True},
        run_options={'shots': 8000, 'rep_delay': 5.e-4},
        # Need the following at the config level for add_readout_mitigation to work properly
        analysis_options={
            'data_processor': DataProcessor('counts', dp_nodes),
            'qutrit_assignment_matrix': assignment_matrix
        }
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cr_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    return unitaries(runner, CrossResonanceGate())


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def rcr_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..calibrations.qutrit_qubit_cx import get_rcr_gate
    sched = get_rcr_gate(runner.qubits, runner.calibrations, target=runner.backend.target)
    return unitaries(runner, RCRGate([]), sched=sched)


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def rcrplus_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    """Unitaries measurement for the RCR in CRCR."""
    calibrations = runner.calibrations

    rcr_type = calibrations.get_parameter_value('rcr_type', runner.qubits)
    x12 = get_qutrit_pulse_gate('x12', runner.qubits[0], calibrations, target=runner.backend.target)
    x_c = calibrations.get_schedule('x', runner.qubits[0])
    x_t = calibrations.get_schedule('x', runner.qubits[1])

    def x_dd():
        with pulse.align_left():
            pulse.call(x_c)
            pulse.call(x_t)
            pulse.call(x_t)

    def x12_dd():
        with pulse.align_left():
            pulse.call(x12)
            pulse.call(x_t)
            pulse.call(x_t)

    target_drive_channel = runner.backend.drive_channel(runner.qubits[1])
    cr = calibrations.get_schedule('cr', runner.qubits)

    # RCR in CRCR
    with pulse.build(name='rcr', default_alignment='sequential') as sched:
        if rcr_type == 2:
            x12_dd()
            pulse.call(cr)
            x_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.call(cr)
        else:
            pulse.call(cr)
            x12_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.call(cr)
            x_dd()

    return unitaries(runner, RCRGate([]), sched=sched)


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def rcrminus_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    """Unitaries measurement for the RCR in CRCR."""
    calibrations = runner.calibrations

    rcr_type = calibrations.get_parameter_value('rcr_type', runner.qubits)
    x12 = get_qutrit_pulse_gate('x12', runner.qubits[0], calibrations, target=runner.backend.target)
    x_c = calibrations.get_schedule('x', runner.qubits[0])
    x_t = calibrations.get_schedule('x', runner.qubits[1])

    def x_dd():
        with pulse.align_left():
            pulse.call(x_c)
            pulse.call(x_t)
            pulse.call(x_t)

    def x12_dd():
        with pulse.align_left():
            pulse.call(x12)
            pulse.call(x_t)
            pulse.call(x_t)

    target_drive_channel = runner.backend.drive_channel(runner.qubits[1])
    control_channel = runner.backend.control_channel(runner.qubits)[0]
    cr = calibrations.get_schedule('cr', runner.qubits)

    with pulse.build(name='rcr', default_alignment='sequential') as sched:
        with pulse.phase_offset(np.pi, control_channel):
            if rcr_type == 2:
                x12_dd()
                with pulse.phase_offset(np.pi, target_drive_channel):
                    pulse.call(cr)
                x_dd()
                pulse.call(cr)
            else:
                with pulse.phase_offset(np.pi, target_drive_channel):
                    pulse.call(cr)
                x12_dd()
                pulse.call(cr)
                x_dd()

    return unitaries(runner, RCRGate([]), sched=sched)


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def crcr_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..calibrations.qutrit_qubit_cx import get_crcr_gate
    sched = get_crcr_gate(runner.qubits, runner.calibrations, target=runner.backend.target)
    return unitaries(runner, CRCRGate(), sched=sched)


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cx_unitaries(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..calibrations.qutrit_qubit_cx import get_qutrit_qubit_cx_gate
    sched = get_qutrit_qubit_cx_gate(runner.qubits, runner.calibrations,
                                     target=runner.backend.target)
    return unitaries(runner, QutritQubitCXGate(), sched=sched)


@register_exp
def cr_initial_amp(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.cr_initial_amp import CRInitialAmplitudeCal
    assignment_matrix = runner.program_data['qutrit_assignment_matrix'][runner.qubits[0]]
    return ExperimentConfig(
        CRInitialAmplitudeCal,
        runner.qubits,
        analysis_options={'assignment_matrix': assignment_matrix}
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1], expval=True)
def cr_rough_width(runner: ExperimentsRunner) -> ExperimentConfig:
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
def cr_angle(runner: ExperimentsRunner) -> ExperimentConfig:
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
def cr_counter_stark_amp(runner: ExperimentsRunner) -> ExperimentConfig:
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
def rcr_rough_cr_amp(runner: ExperimentsRunner) -> ExperimentConfig:
    """CR angle calibration to eliminate the y component of RCR non-participating state."""
    from ..experiments.qutrit_qubit_cx.cr_amp import CRRoughAmplitudeCal

    def cal_criterion(data):
        return data.analysis_results('pi_amp', block=False).value.n < 0.95

    return ExperimentConfig(
        CRRoughAmplitudeCal,
        runner.qubits,
        calibration_criterion=cal_criterion
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1])
def rcr_rotary_amp(runner: ExperimentsRunner) -> ExperimentConfig:
    """Rotary tone amplitude calibration to minimize the y and z components of RCR."""
    from ..experiments.qutrit_qubit_cx.rotary import RepeatedCRRotaryAmplitudeCal
    duration = gs_effective_duration(runner.calibrations, runner.qubits, 'cr')
    cycles_per_amp = rabi_cycles_per_area(runner.backend, runner.qubits[1]) * duration
    cycles_per_amp *= 2  # Factor two because RCR = CR * 2
    amplitudes = np.linspace(0.5, 2.5, 20) / cycles_per_amp
    return ExperimentConfig(
        RepeatedCRRotaryAmplitudeCal,
        runner.qubits,
        args={'amplitudes': amplitudes},
        analysis_options={'parallelize': -1, 'parallelize_on_thread': True}
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1])
def crcr_fine_scanbased(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit_cx.crcr_fine import CycledRepeatedCRFineScanCal
    return ExperimentConfig(
        CycledRepeatedCRFineScanCal,
        runner.qubits
    )


@register_exp
@add_readout_mitigation(logical_qubits=[1])
def tc2_cr_rotary_delta(runner: ExperimentsRunner) -> ExperimentConfig:
    from ..experiments.qutrit_qubit.rotary_stark_shift import RotaryStarkShiftPhaseCal
    return ExperimentConfig(
        RotaryStarkShiftPhaseCal,
        tuple(reversed(runner.qubits)),
        analysis_options={'outcome': '1'}
    )
