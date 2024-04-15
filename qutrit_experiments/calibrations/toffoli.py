"""Functions to define calibrations for the Toffoli gate."""

from collections.abc import Sequence
import logging
from typing import Optional
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

from ..pulse_library import DoubleDrag
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_operational_qubits

logger = logging.getLogger(__name__)

def make_toffoli_calibrations(
    backend: Backend,
    calibrations: Calibrations,
    set_defaults: bool = True,
    qubits: Optional[Sequence[int]] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    add_dd(calibrations)

    if set_defaults:
        set_dd_default(backend, calibrations, qubits)

    return calibrations


def add_dd(
    calibrations: Calibrations
) -> None:
    """Add templates for DD sequences (X-delay-X)."""
    duration = Parameter('duration')
    pulse_duration = Parameter('pulse_duration')

    with pulse.build(name='dd_left') as sched:
        pulse.play(DoubleDrag(duration=duration,
                              pulse_duration=pulse_duration,
                              amp=Parameter('amp'),
                              sigma=Parameter('sigma'),
                              beta=Parameter('beta'),
                              interval=(0, (duration - 2 * pulse_duration) / 2),
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
                              interval=((duration - 2 * pulse_duration) / 2, 0),
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
