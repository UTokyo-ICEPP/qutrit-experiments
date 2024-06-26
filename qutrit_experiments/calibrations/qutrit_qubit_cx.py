"""Functions to generate the Calibrations object for qutrit experiments."""

from collections.abc import Sequence
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

from ..constants import LO_SIGN
from ..gates import ParameterValueType
from ..pulse_library import ModulatedGaussianSquare
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_operational_qubits, get_qutrit_qubit_composite_gate
from .qubit import add_x, set_x_default

LOG = logging.getLogger(__name__)


def make_qutrit_qubit_cx_calibrations(
    backend: Backend,
    calibrations: Optional[Calibrations] = None,
    set_defaults: bool = True,
    qubits: Optional[Sequence[int]] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    add_qutrit_qubit_cr(calibrations)
    if not calibrations.has_template('x'):
        add_x(calibrations)
    add_qutrit_qubit_rcr(calibrations)
    add_qutrit_qubit_cx(calibrations)

    calibrations._register_parameter(Parameter('rcr_type'), ())
    calibrations._register_parameter(Parameter('delta_rzx45p_rotary'), ())

    if set_defaults:
        set_qutrit_qubit_cr_default(backend, calibrations, qubits=qubits)
        set_x_default(backend, calibrations, qubits=qubits)
        instantiate_qutrit_qubit_cx(backend, calibrations, qubits=qubits)

    return calibrations


def add_qutrit_qubit_cr(
    calibrations: Calibrations
) -> None:
    """Define CR and counter tones with ModulatedGaussianSquare potentiallly with an additional
    off-resonant component.

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    rsr = Parameter('rsr')
    sigma = Parameter('sigma')
    width = Parameter('width')
    margin = Parameter('margin')
    duration = sigma * rsr * 2 + width + margin
    stark_detuning = Parameter('stark_freq') - Parameter('target_freq')

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
                                       angle=cr_base_angle + cr_sign_angle,
                                       risefall_sigma_ratio=rsr, phases=(cr_stark_sign_phase,),
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


def set_qutrit_qubit_cr_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Give default values to CR parameters.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
    try:
        twoq_target = backend.target['cx']
    except KeyError:
        twoq_target = backend.target['ecr']

    for gate_qubits, gate_spec in twoq_target.items():
        if not set(gate_qubits) <= operational_qubits:
            continue

        twoq_sched = gate_spec.calibration
        control_channel = backend.control_channel(gate_qubits)[0]
        control_instructions = [inst for _, inst in twoq_sched.instructions
                                if isinstance(inst, pulse.Play) and inst.channel == control_channel]
        if not control_instructions:
            # Reverse CR
            continue

        try:
            default_cr = next(inst.pulse for inst in control_instructions
                              if inst.name.startswith('CR90p'))
        except StopIteration:  # Direct CX
            try:
                default_cr = next(inst.pulse for inst in control_instructions
                                  if inst.name.startswith('CX'))
            except StopIteration as exc:
                raise CalibrationError(
                    f'No default CR instruction for qubits {gate_qubits}'
                ) from exc

        target_freq = backend.qubit_properties(gate_qubits[0]).frequency * backend.dt

        param_defaults = [
            ('rsr', (default_cr.duration - default_cr.width) / 2. / default_cr.sigma),
            ('sigma', default_cr.sigma),
            ('width', 0.),
            ('margin', 0.),
            ('stark_freq', target_freq),
            ('target_freq', target_freq),
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
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=gate_qubits,
                                             schedule='cr')

        calibrations.add_parameter_value(ParameterValue(0.), 'delta_rzx45p_rotary',
                                         qubits=gate_qubits)


def add_qutrit_qubit_rcr(
    calibrations: Calibrations
) -> None:
    """Add the RCR and CX schedules.

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    control_channel = pulse.ControlChannel(Parameter('ch0.1'))
    target_drive_channel = pulse.DriveChannel(Parameter('ch1'))

    # X/X12 on qubit 0 with DD on qubit 1
    def x_dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q1')
            pulse.reference('x', 'q1')

    def x12_dd():
        with pulse.align_left():
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
        x12_dd()
        with pulse.phase_offset(np.pi, target_drive_channel):
            pulse.reference('cr', 'q0', 'q1')
        x12_dd()
    calibrations.add_schedule(sched, num_qubits=2)

    # CRCR type X
    with pulse.build(name='crcr2', default_alignment='sequential') as sched:
        # [X12+DD][CR-][X+DD][CR-]
        with pulse.phase_offset(np.pi, control_channel):
            x12_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x_dd()
            pulse.reference('cr', 'q0', 'q1')
        for _ in range(2):
            # [X12+DD][CR+][X+DD][CR+]
            x12_dd()
            pulse.reference('cr', 'q0', 'q1')
            x_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
    calibrations.add_schedule(sched, num_qubits=2)

    # CRCR type X12
    with pulse.build(name='crcr0', default_alignment='sequential') as sched:
        for _ in range(2):
            # [CR+][X12+DD][CR+][X+DD]
            pulse.reference('cr', 'q0', 'q1')
            x12_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x_dd()
        # [CR-][X12+DD][CR-][X+DD]
        with pulse.phase_offset(np.pi, control_channel):
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x12_dd()
            pulse.reference('cr', 'q0', 'q1')
            x_dd()
    calibrations.add_schedule(sched, num_qubits=2)


def add_qutrit_qubit_cx(
    calibrations: Calibrations
) -> None:
    """Add unbound schedule for qutrit-qubit CX.

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    # Offset Rx for CX (template)
    duration = Parameter('duration')
    amp = Parameter('amp')
    sigma = Parameter('sigma')
    beta = Parameter('beta')
    base_angle = Parameter('base_angle')
    angle = Parameter('angle')
    drive_channel = pulse.DriveChannel(Parameter('ch0'))
    with pulse.build(name='cx_offset_rx', default_alignment='sequential') as rx_sched:
        pulse.play(
            pulse.Drag(duration=duration, amp=amp, sigma=sigma,
                       beta=beta, angle=base_angle + LO_SIGN * np.pi / 2.,
                       name='SYp'),
            drive_channel,
            name='SYp'
        )
        pulse.play(
            pulse.Drag(duration=duration, amp=amp, sigma=sigma,
                       beta=beta, angle=base_angle + LO_SIGN * (angle + 1.5 * np.pi),
                       name='SΘp'),
            drive_channel,
            name='SΘp'
        )
        # Don't add the rz label; target is operated as a qubit
        pulse.shift_phase(LO_SIGN * angle, drive_channel)
    calibrations.add_schedule(rx_sched, num_qubits=1)

    # Geometric phase correction (template)
    cx_sign = Parameter('cx_sign')
    with pulse.build(name='cx_geometric_phase') as geom_sched:
        # pulse.shift_phase(LO_SIGN * cx_sign * np.pi / 3., drive_channel, name='rz')
        # pulse.shift_phase(LO_SIGN * cx_sign * np.pi / 6., drive_channel, name='rz12')
        #  The transpiler creates -LO_SIGN * cx_sign * pi/2 EF phase from above two lines. This
        #  implementation, although with a clearer physics picture, is kind of stupid, so we instead
        #  consolidate the lines to a single pi/2 GE phase shift + -pi/2 EF phase shift, where the
        #  second line below will be removed from the schedule by the transpiler.
        pulse.shift_phase(LO_SIGN * cx_sign * np.pi / 2., drive_channel)
        pulse.shift_phase(-LO_SIGN * cx_sign * np.pi / 2., drive_channel, name='ef_phase')
    calibrations.add_schedule(geom_sched, num_qubits=1)

    # Calibrations.get_schedule cannot assign parameters to nested references so we repeat the
    # CRCR implementations here instead of referencing them
    control_channel = pulse.ControlChannel(Parameter('ch0.1'))
    target_drive_channel = pulse.DriveChannel(Parameter('ch1'))

    # X/X12 on qubit 0 with DD on qubit 1
    def x_dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q1')
            pulse.reference('x', 'q1')

    def x12_dd():
        with pulse.align_left():
            pulse.reference('x12', 'q0')
            pulse.reference('x', 'q1')
            pulse.reference('x', 'q1')

    # CX type X
    with pulse.build(name='qutrit_qubit_cx_rcr2', default_alignment='sequential') as sched:
        # [X12+DD][CR-][X+DD][CR-]
        with pulse.align_left():
            pulse.reference('x12', 'q0')
            pulse.reference('cx_offset_rx', 'q1')
        with pulse.phase_offset(np.pi, control_channel):
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x_dd()
            pulse.reference('cr', 'q0', 'q1')
        for _ in range(2):
            # [X12+DD][CR+][X+DD][CR+]
            x12_dd()
            pulse.reference('cr', 'q0', 'q1')
            x_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
        pulse.reference('cx_geometric_phase', 'q0')
    calibrations.add_schedule(sched, num_qubits=2)

    # CX type X12
    with pulse.build(name='qutrit_qubit_cx_rcr0', default_alignment='sequential') as sched:
        for _ in range(2):
            # [CR+][X12+DD][CR+][X+DD]
            pulse.reference('cr', 'q0', 'q1')
            x12_dd()
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x_dd()
        # [CR-][X12+DD][CR-][X+DD]
        with pulse.phase_offset(np.pi, control_channel):
            with pulse.phase_offset(np.pi, target_drive_channel):
                pulse.reference('cr', 'q0', 'q1')
            x12_dd()
            pulse.reference('cr', 'q0', 'q1')
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('cx_offset_rx', 'q1')
        pulse.reference('cx_geometric_phase', 'q0')
    calibrations.add_schedule(sched, num_qubits=2)


def instantiate_qutrit_qubit_cx(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Instantiate the cx_offset_rx and cx_geometric_phase schedules for each qubit pair.

    Since the last shift_phases are parts of Rz, which must be applied to variable number of
    channels, we instantiate a full schedule for each qubit.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    operational_qubits = get_operational_qubits(backend, qubits=qubits)

    try:
        twoq_target = backend.target['cx']
    except KeyError:
        twoq_target = backend.target['ecr']

    for gate_qubits in twoq_target.keys():
        if not set(gate_qubits) <= operational_qubits:
            continue

        # Offset Rx
        sx_inst = backend.target['sx'][gate_qubits[1:]].calibration.instructions[0][1]
        sx_pulse = sx_inst.pulse
        drive_channel = sx_inst.channel
        rz_sched = backend.target['rz'][gate_qubits[1:]].calibration
        rz_channels = [inst.channel for _, inst in rz_sched.instructions]
        rz_channels.remove(drive_channel)  # The template already has drive_channel

        rx_sched = calibrations.get_template('cx_offset_rx')
        angle = rx_sched.get_parameters('angle')[0]
        phase = LO_SIGN * angle
        qubit_sched = rx_sched
        for channel in rz_channels:
            # Managing backend configuration error: Seen a case where a ControlChannel included in
            # the target calibration instructions is actually not valid in terms of coupling_map
            if channel not in calibrations._controls_config_r:
                continue
            qubit_sched = qubit_sched.append(pulse.ShiftPhase(phase, channel), inplace=False)
        calibrations.add_schedule(qubit_sched, qubits=gate_qubits[1:])

        param_defaults = [
            ('duration', sx_pulse.duration),
            ('amp', sx_pulse.amp),
            ('sigma', sx_pulse.sigma),
            ('beta', sx_pulse.beta),
            ('base_angle', sx_pulse.angle)
        ]
        for pname, value in param_defaults:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=gate_qubits[1:],
                                             schedule=rx_sched.name)
        calibrations.add_parameter_value(ParameterValue(0.), angle.name, qubits=gate_qubits[1:],
                                         schedule=rx_sched.name)

        # Geometric phase
        drive_channel = backend.target['x'][gate_qubits[:1]].calibration.instructions[0][1].channel
        rz_sched = backend.target['rz'][gate_qubits[:1]].calibration
        rz_channels = [inst.channel for _, inst in rz_sched.instructions]
        rz_channels.remove(drive_channel)  # Already in the template

        geom_sched = calibrations.get_template('cx_geometric_phase')
        cx_sign = geom_sched.get_parameters('cx_sign')[0]
        phase = LO_SIGN * cx_sign * np.pi / 2.
        qubit_sched = geom_sched
        for channel in rz_channels:
            # See above
            if channel not in calibrations._controls_config_r:
                continue
            qubit_sched = qubit_sched.append(pulse.ShiftPhase(phase, channel), inplace=False)
        calibrations.add_schedule(qubit_sched, qubits=gate_qubits[:1])

        calibrations.add_parameter_value(ParameterValue(0), cx_sign.name, qubits=gate_qubits[:1],
                                         schedule=geom_sched.name)


def get_rcr_gate(
    physical_qubits: Sequence[int],
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    """Return the RCR gate according to the rcr_type of the qutrit-qubit system.

    Args:
        physical_qubits: (qutrit, qubit)
        calibrations: Calibrations object.
        freq_shift: Frequency difference between EF and GE transitions, in 1/dt as returned by
            ``.util.get_qutrit_freq_shift()``.
        target: Backend transpiler target.
        assign_params: Optional parameter values to assign.
        group: Calibration group name.
    """
    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
    return get_qutrit_qubit_composite_gate(f'rcr{rcr_type}', physical_qubits, calibrations,
                                           freq_shift=freq_shift, target=target,
                                           assign_params=assign_params, group=group)


def get_crcr_gate(
    physical_qubits: Sequence[int],
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    """Return the CRCR gate according to the rcr_type of the qutrit-qubit system.

    Args:
        physical_qubits: (qutrit, qubit)
        calibrations: Calibrations object.
        freq_shift: Frequency difference between EF and GE transitions, in 1/dt as returned by
            ``.util.get_qutrit_freq_shift()``.
        target: Backend transpiler target.
        assign_params: Optional parameter values to assign.
        group: Calibration group name.
    """
    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
    return get_qutrit_qubit_composite_gate(f'crcr{rcr_type}', physical_qubits, calibrations,
                                           freq_shift=freq_shift, target=target,
                                           assign_params=assign_params, group=group)


def get_qutrit_qubit_cx_gate(
    physical_qubits: Sequence[int],
    calibrations: Calibrations,
    freq_shift: Optional[float] = None,
    target: Optional[Target] = None,
    assign_params: Optional[dict[str, ParameterValueType]] = None,
    group: str = 'default'
) -> ScheduleBlock:
    """Return the Qutrit-qubit generalized CX gate according to the rcr_type.

    Args:
        physical_qubits: (qutrit, qubit)
        calibrations: Calibrations object.
        freq_shift: Frequency difference between EF and GE transitions, in 1/dt as returned by
            ``.util.get_qutrit_freq_shift()``.
        target: Backend transpiler target.
        assign_params: Optional parameter values to assign.
        group: Calibration group name.
    """
    rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
    return get_qutrit_qubit_composite_gate(f'qutrit_qubit_cx_rcr{rcr_type}', physical_qubits,
                                           calibrations, freq_shift=freq_shift, target=target,
                                           assign_params=assign_params, group=group)
