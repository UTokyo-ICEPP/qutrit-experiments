from collections.abc import Sequence
import logging
from typing import Optional
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from ..pulse_library import DoubleDrag
from .util import get_operational_qubits

logger = logging.getLogger(__name__)


def add_x(
    calibrations: Calibrations
) -> None:
    """Add the X schedule to copy the parameters from backend defaults.

    Note: Defining a template and binding constants from inst_map seems to unnecessarily increase
    the recall overhead of the schedule (plus we lose the pulse names specific to the drive
    channels). However, for the unbound compound schedules to be able to reference this schedule,
    it needs to be defined unbounded.
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
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)

    # Import the parameter values from inst_map
    for qubit in operational_qubits:
        params = inst_map.get('x', qubit).instructions[0][1].pulse.parameters
        for pname, value in params.items():
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule='x')


def add_dd(
    calibrations: Calibrations
) -> None:
    """Add templates for DD sequences (X-delay-X)."""
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
    """Copy the backend defaults X parameter values into DoubleDrag."""
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
    for qubit in operational_qubits:
        x_sched = inst_map.get('x', qubit)
        x_pulse = x_sched.instructions[0][1].pulse
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
