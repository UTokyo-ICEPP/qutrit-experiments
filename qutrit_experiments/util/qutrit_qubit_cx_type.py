"""Utility functions for qutrit Toffoli."""
from collections.abc import Sequence
from qiskit.providers import Backend
from qiskit.providers.exceptions import BackendConfigurationError
from qiskit.pulse import ControlChannel

from ..calibrations.util import get_default_ecr_schedule
from ..gates import QutritQubitCXType


def qutrit_qubit_cx_type(
    backend: Backend,
    physical_qubits: Sequence[int]
) -> int:
    """Return the cx type to use according to the default cx schedule.

    This function is not limited to Toffoli and should be moved to somewhere else
    """
    physical_qubits = tuple(physical_qubits)
    try:
        control_channel = backend.control_channel(physical_qubits)
    except BackendConfigurationError:
        return QutritQubitCXType.REVERSE

    ecr_sched = get_default_ecr_schedule(backend, physical_qubits)
    cr_inst = next(inst for _, inst in ecr_sched.instructions
                   if isinstance(inst.channel, ControlChannel))
    if cr_inst.channel == control_channel:
        return QutritQubitCXType.UNKNOWN
    return QutritQubitCXType.REVERSE
