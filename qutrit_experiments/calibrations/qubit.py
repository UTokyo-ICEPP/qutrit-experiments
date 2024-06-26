"""
Functions for defining single-qubit calibrations.
"""
from collections.abc import Sequence
import logging
from typing import Optional
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScalableSymbolicPulse
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from ..pulse_library import DoubleDrag
from .util import get_operational_qubits

LOG = logging.getLogger(__name__)


def add_x(
    calibrations: Calibrations
) -> None:
    """Add the X schedule to copy the parameters from backend defaults.

    Note: Defining a template and binding constants from inst_map seems to unnecessarily increase
    the recall overhead of the schedule (plus we lose the pulse names specific to the drive
    channels). However, for the unbound compound schedules to be able to reference this schedule,
    it needs to be defined unbounded.

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    with pulse.build(name='x') as sched:
        pulse.play(
            pulse.Drag(duration=Parameter('duration'), amp=Parameter('amp'),
                       sigma=Parameter('sigma'), beta=Parameter('beta'), angle=Parameter('angle'),
                       name='Xp'),
            pulse.DriveChannel(Parameter('ch0')),
            name='Xp'
        )
    calibrations.add_schedule(sched, num_qubits=1)


def set_x_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Set the default parameter values for the X schedule.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)

    # Import the parameter values from inst_map
    for qubit in operational_qubits:
        x_pulse = inst_map.get('x', qubit).instructions[0][1].pulse
        if not (isinstance(x_pulse, ScalableSymbolicPulse)
                and x_pulse.pulse_type == 'Drag'):
            raise RuntimeError(f'Pulse of q{qubit} x is not Drag')
        for pname, value in x_pulse.parameters.items():
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='x')


def add_dd(
    calibrations: Calibrations
) -> None:
    """Add templates for DD sequences (X-delay-X).

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    duration = Parameter('duration')
    pulse_duration = Parameter('pulse_duration')
    interval = (duration - 2 * pulse_duration) / 2

    with pulse.build(name='dd_left') as sched:
        pulse.play(DoubleDrag(duration=duration,
                              pulse_duration=pulse_duration,
                              amp=Parameter('amp'),
                              sigma=Parameter('sigma'),
                              beta=Parameter('beta'),
                              interval=(0, interval),
                              name='DD'),
                   pulse.DriveChannel(Parameter('ch0')),
                   name='DD')
    calibrations.add_schedule(sched, num_qubits=1)

    with pulse.build(name='dd_right') as sched:
        pulse.play(DoubleDrag(duration=duration,
                              pulse_duration=pulse_duration,
                              amp=Parameter('amp'),
                              sigma=Parameter('sigma'),
                              beta=Parameter('beta'),
                              interval=(interval, 0),
                              name='DD'),
                   pulse.DriveChannel(Parameter('ch0')),
                   name='DD')
    calibrations.add_schedule(sched, num_qubits=1)

    with pulse.build(name='dd_symmetric') as sched:
        pulse.play(DoubleDrag(duration=duration,
                              pulse_duration=pulse_duration,
                              amp=Parameter('amp'),
                              sigma=Parameter('sigma'),
                              beta=Parameter('beta'),
                              interval=(interval / 2, interval),
                              name='DD'),
                   pulse.DriveChannel(Parameter('ch0')),
                   name='DD')
    calibrations.add_schedule(sched, num_qubits=1)


def set_dd_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Copy the backend defaults X parameter values into DoubleDrag.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
    for qubit in operational_qubits:
        x_sched = inst_map.get('x', qubit)
        x_pulse = x_sched.instructions[0][1].pulse
        if not (isinstance(x_pulse, ScalableSymbolicPulse)
                and x_pulse.pulse_type == 'Drag'):
            raise RuntimeError(f'Pulse of q{qubit} x is not Drag')
        x_duration = x_sched.duration
        pvalues = [
            ('duration', x_duration * 2),
            ('pulse_duration', x_pulse.duration),
            ('amp', x_pulse.amp),
            ('sigma', x_pulse.sigma),
            ('beta', x_pulse.beta)
        ]
        for pname, value in pvalues:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='dd_left')
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='dd_right')
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='dd_symmetric')


def add_ecr(
    calibrations: Calibrations
) -> None:
    """Add the ECR schedule for CX-based backends.

    Args:
        calibrations: Calibrations to define the unbound schedule in.
    """
    duration = Parameter('duration')
    sigma = Parameter('sigma')
    width = Parameter('width')
    cr_amp = Parameter('cr_amp')
    cr_angle = Parameter('cr_angle')
    rotary_amp = Parameter('rotary_amp')
    rotary_angle = Parameter('rotary_angle')
    control_channel = pulse.ControlChannel(Parameter('ch0.1'))
    drive_channel = pulse.DriveChannel(Parameter('ch1'))

    with pulse.build(name='ecr', default_alignment='sequential') as sched:
        with pulse.align_left():
            pulse.play(pulse.GaussianSquare(duration=duration, sigma=sigma, width=width, amp=cr_amp,
                                            angle=cr_angle, name='CR90p_u'),
                       control_channel, name='CR90p_u')
            pulse.play(pulse.GaussianSquare(duration=duration, sigma=sigma, width=width,
                                            amp=rotary_amp,
                                            angle=rotary_angle, name='CR90p_d'),
                       drive_channel, name='CR90p_d')
        pulse.reference('x', 'q0')
        with pulse.align_left():
            pulse.play(pulse.GaussianSquare(duration=duration, sigma=sigma, width=width, amp=cr_amp,
                                            angle=cr_angle + np.pi, name='CR90m_u'),
                       control_channel, name='CR90m_u')
            pulse.play(pulse.GaussianSquare(duration=duration, sigma=sigma, width=width,
                                            amp=rotary_amp,
                                            angle=rotary_angle + np.pi, name='CR90m_d'),
                       drive_channel,
                       name='CR90m_d')
    calibrations.add_schedule(sched, num_qubits=2)


def set_ecr_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Copy the background defaults CX parameter values into ECR.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
    for qubits, cx_spec in backend.target['cx'].items():
        if not set(qubits) <= operational_qubits:
            continue

        cx_sched = cx_spec.calibration
        try:
            cr_pulse = next(inst.pulse for _, inst in cx_sched.instructions
                            if isinstance(inst, pulse.Play)
                            and isinstance(inst.channel, pulse.ControlChannel)
                            and inst.name.startswith('CR90p_u'))
            rotary_pulse = next(inst.pulse for _, inst in cx_sched.instructions
                                if isinstance(inst, pulse.Play)
                                and isinstance(inst.channel, pulse.DriveChannel)
                                and inst.name.startswith('CR90p_d'))
        except StopIteration as exc:
            raise CalibrationError(f'Default ECR CR pulse not found for qubits {qubits}.') from exc

        pvalues = [
            ('duration', cr_pulse.duration),
            ('sigma', cr_pulse.sigma),
            ('width', cr_pulse.width),
            ('cr_amp', cr_pulse.amp),
            ('cr_angle', cr_pulse.angle),
            ('rotary_amp', rotary_pulse.amp),
            ('rotary_angle', rotary_pulse.angle)
        ]
        for pname, value in pvalues:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=qubits,
                                             schedule='ecr')
