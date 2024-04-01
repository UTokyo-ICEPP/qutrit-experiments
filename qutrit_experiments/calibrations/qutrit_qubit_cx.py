"""Functions to generate the Calibrations object for qutrit experiments."""

import logging
from typing import Optional
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations, ParameterValue
from qiskit_experiments.exceptions import CalibrationError

from ..gates import ParameterValueType
from ..pulse_library import ModulatedGaussianSquare
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_default_ecr_schedule, get_qutrit_freq_shift

logger = logging.getLogger(__name__)


def make_qutrit_qubit_cx_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    add_qutrit_qubit_cr(backend, calibrations)
    add_qutrit_qubit_cx(backend, calibrations)

    calibrations._register_parameter(Parameter('rcr_type'), ())
    calibrations._register_parameter(Parameter('qutrit_qubit_cx_sign'), ())

    return calibrations


def add_qutrit_qubit_cr(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    """Define CR and counter tones with ModulatedGaussianSquare potentiallly with an additional
    off-resonant component."""
    rsr = Parameter('rsr')
    sigma = Parameter('sigma')
    width = Parameter('width')
    margin = Parameter('margin')
    duration = sigma * rsr * 2 + width + margin
    stark_frequency = Parameter('stark_frequency')
    stark_detuning = (stark_frequency - Parameter('target_frequency')) * backend.dt

    cr_amp = Parameter('cr_amp')
    cr_base_angle = Parameter('cr_base_angle')
    cr_sign_angle = Parameter('cr_sign_angle')
    cr_stark_amp = Parameter('cr_stark_amp')
    cr_stark_sign_phase = Parameter('cr_stark_sign_phase')
    cr_full_amp = cr_amp + cr_stark_amp

    counter_amp = Parameter('counter_amp')
    counter_base_angle = Parameter('counter_base_angle')
    counter_sign_angle = Parameter('counter_sign_angle')
    counter_stark_amp = Parameter('counter_stark_amp')
    counter_full_amp = counter_amp + counter_stark_amp

    cr_pulse = ModulatedGaussianSquare(duration=duration, amp=cr_full_amp,
                                       sigma=sigma, freq=(0., stark_detuning),
                                       fractions=(cr_amp, cr_stark_amp), width=width,
                                       angle=cr_base_angle + cr_sign_angle, risefall_sigma_ratio=rsr,
                                       phases=(cr_stark_sign_phase,),
                                       name='CR')
    counter_pulse = ModulatedGaussianSquare(duration=duration, amp=counter_full_amp,
                                            sigma=sigma, freq=(0., stark_detuning),
                                            fractions=(counter_amp, counter_stark_amp), width=width,
                                            angle=counter_base_angle + counter_sign_angle,
                                            risefall_sigma_ratio=rsr, name='Counter')

    with pulse.build(name='cr', default_alignment='left') as sched:
        pulse.play(cr_pulse, pulse.ControlChannel(Parameter('ch0.1')), name='CR')
        pulse.play(counter_pulse, pulse.DriveChannel(Parameter('ch1')), name='Counter')
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits, control_channels in backend.control_channels.items():
        control_channel = control_channels[0]
        target_frequency = backend.qubit_properties(qubits[1]).frequency
        ecr_sched = get_default_ecr_schedule(backend, qubits)
        control_instructions = [inst for _, inst in ecr_sched.instructions
                                if isinstance(inst, pulse.Play) and inst.channel == control_channel]
        if not control_instructions:
            # Reverse CR
            continue
        try:
            default_cr = next(inst.pulse for inst in control_instructions
                              if inst.name.startswith('CR90p'))
        except StopIteration: # Direct CX
            try:
                default_cr = next(inst.pulse for inst in control_instructions
                                  if inst.name.startswith('CX'))
            except StopIteration as exc:
                raise CalibrationError(f'No default CR instruction for qubits {qubits}') from exc

        param_defaults = [
            ('rsr', (default_cr.duration - default_cr.width) / 2. / default_cr.sigma),
            ('sigma', default_cr.sigma),
            ('width', 0.),
            ('margin', 0.),
            ('stark_frequency', target_frequency),
            ('target_frequency', target_frequency),
            ('cr_amp', 0.),
            ('cr_base_angle', default_cr.angle),
            ('cr_sign_angle', 0.),
            ('cr_stark_amp', 0.),
            ('cr_stark_sign_phase', 0.),
            ('counter_amp', 0.),
            ('counter_base_angle', 0.),
            ('counter_sign_angle', 0.),
            ('counter_stark_amp', 0.)
        ]
        for pname, value in param_defaults:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=qubits,
                                             schedule='cr')


def add_qutrit_qubit_cx(
    backend: Backend,
    calibrations: Calibrations
) -> None:
    """Add the RCR and CX schedules. Also import the X schedule from default inst_map."""
    # X schedule
    with pulse.build(name='x') as sched:
        pulse.play(
            pulse.Drag(duration=Parameter('duration'), amp=Parameter('amp'),
                       sigma=Parameter('sigma'), beta=Parameter('beta'), angle=Parameter('angle'),
                       name='X'),
            pulse.DriveChannel(Parameter('ch0'))
        )
    calibrations.add_schedule(sched, num_qubits=1)

    # Import the parameter values from inst_map
    inst_map = backend.defaults().instruction_schedule_map
    for qubit in range(backend.num_qubits):
        params = inst_map.get('x', qubit).instructions[0][1].pulse.parameters
        for pname, value in params.items():
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='x')
            
    control_channel = pulse.ControlChannel(Parameter('ch0.1'))
    control_drive_channel = pulse.DriveChannel(Parameter('ch0'))
    target_drive_channel = pulse.DriveChannel(Parameter('ch1'))

    # X/X12 on qubit 0 with DD on qubit 1
    def x_dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q1')
            pulse.reference('x', 'q1')

    def x12_dd(idx):
        with pulse.align_left():
            with pulse.phase_offset(Parameter(f'ef_phase_{idx}'), control_drive_channel):
                pulse.reference('x12', 'q0')
                pulse.reference('x', 'q1')
                pulse.reference('x', 'q1')

    # RCR type X
    with pulse.build(name='rcr2', default_alignment='sequential') as sched:
        x_dd()
        pulse.reference('cr', 'q0', 'q1')
        x_dd()
        with pulse.phase_offset(np.pi, target_drive_channel):
            pulse.reference('cr', 'q0', 'q1')
    calibrations.add_schedule(sched, num_qubits=2)

    # RCR type X12
    with pulse.build(name='rcr0', default_alignment='sequential') as sched:
        pulse.reference('cr', 'q0', 'q1')
        x12_dd(0)
        with pulse.phase_offset(np.pi, target_drive_channel):
            pulse.reference('cr', 'q0', 'q1')
        x12_dd(1)
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits in backend.control_channels.keys():
        for irep in range(3):
            calibrations.add_parameter_value(ParameterValue(0.), f'ef_phase_{irep}', qubits=qubits,
                                             schedule=sched.name)

    # CRCR requires an offset Rx, which involves an Rz gate.
    # Since the number of channels in an Rz instruction varies qubit by qubit, we have to
    # instantiate a full schedule for each qubit.
    angle = Parameter('angle')
    for _, qubit in backend.control_channels.keys():
        sx_inst = inst_map.get('sx', qubit).instructions[0][1]
        sx_pulse = sx_inst.pulse
        drive_channel = sx_inst.channel
        rz_channels = [inst.channel for _, inst in inst_map.get('rz', qubit).instructions]
        with pulse.build(name='cx_offset_rx', default_alignment='sequential') as sched:
            pulse.play(
                pulse.Drag(duration=sx_pulse.duration, amp=sx_pulse.amp, sigma=sx_pulse.sigma,
                           beta=sx_pulse.beta, angle=sx_pulse.angle - np.pi / 2.,
                           name='SY'),
                drive_channel
            )
            pulse.play(
                pulse.Drag(duration=sx_pulse.duration, amp=sx_pulse.amp, sigma=sx_pulse.sigma,
                           beta=sx_pulse.beta,  angle=sx_pulse.angle - 3. * np.pi / 2. - angle,
                           name='SΘ'),
                drive_channel
            )
            for channel in rz_channels:
                pulse.shift_phase(-angle, channel)
        calibrations.add_schedule(sched, qubits=[qubit])

        calibrations.add_parameter_value(ParameterValue(0.), angle.name, qubits=qubit,
                                         schedule=sched.name)

    # CX type X
    with pulse.build(name='qutrit_qubit_cx_rcr2', default_alignment='sequential') as sched:
        # [X12+Rx][CR-][X+DD][CR-]
        with pulse.align_left():
            with pulse.phase_offset(Parameter('ef_phase_0'), control_drive_channel):
                pulse.reference('x12', 'q0')
            pulse.reference('cx_offset_rx', 'q1')
        with pulse.phase_offset(np.pi, control_channel, target_drive_channel):
            pulse.reference('cr', 'q0', 'q1')
        x_dd()
        with pulse.phase_offset(np.pi, control_channel):
            pulse.reference('cr', 'q0', 'q1')
        for irep in range(2):
            # [X12+DD][CR+][X+DD][CR+]
            x12_dd(irep + 1)
            pulse.reference('cr', 'q0', 'q1')
            x_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits in backend.control_channels.keys():
        for irep in range(3):
            calibrations.add_parameter_value(ParameterValue(0.), f'ef_phase_{irep}', qubits=qubits,
                                             schedule=sched.name)

    with pulse.build(name='qutrit_qubit_cx_rcr0', default_alignment='sequential') as sched:
        for irep in range(2):
            # [CR+][X12+DD][CR+][X+DD]
            pulse.reference('cr', 'q0', 'q1')
            x12_dd(irep)
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x_dd()
        # [CR-][X12+DD][CR-][X+Rx]
        with pulse.phase_offset(np.pi, control_channel, target_drive_channel):
            pulse.reference('cr', 'q0', 'q1')
        x12_dd(2)
        with pulse.phase_offset(np.pi, control_channel):
            pulse.reference('cr', 'q0', 'q1')
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('cx_offset_rx', 'q1')
    calibrations.add_schedule(sched, num_qubits=2)

    for qubits in backend.control_channels.keys():
        for irep in range(3):
            calibrations.add_parameter_value(ParameterValue(0.), f'ef_phase_{irep}', qubits=qubits,
                                             schedule=sched.name)


def get_qutrit_qubit_composite_gate(
    gate_name: str,
    physical_qubits: tuple[int, int],
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    physical_qubits = tuple(physical_qubits)
    template = calibrations.get_template(gate_name, physical_qubits)
    assign_params_dict = {}
    for ref in template.references.unassigned():
        if (gate := ref[0]) in ['x12', 'sx12']:
            key = ('freq', physical_qubits[:1], gate)
            if not freq_shift:
                freq_shift = get_qutrit_freq_shift(physical_qubits[0], target, calibrations)
            assign_params_dict[key] = freq_shift

    for param in template.parameters:
        if param.name.startswith('ef_phase_'):
            assign_params_dict[param.name] = Parameter(param.name)
    
    if assign_params:
        assign_params_dict.update(assign_params)

    return calibrations.get_schedule(gate_name, physical_qubits, assign_params=assign_params_dict,
                                     group=group)
