from collections.abc import Sequence
import logging
from typing import Optional
from qiskit import pulse
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

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
