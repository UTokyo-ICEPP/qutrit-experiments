"""Common utility functions for calibration definitions."""

from collections.abc import Sequence
from typing import Optional
from qiskit.providers import Backend
from qiskit.pulse import PulseError, Schedule
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations


def get_operational_qubits(
    backend: Backend,
    qubits: Optional[Sequence[int]] = None
) -> set[int]:
    try:
        faulty_qubits = set(backend.properties().faulty_qubits())
    except AttributeError:
        # Simulator
        faulty_qubits = set()
    if not qubits:
        qubits = range(backend.num_qubits)
    return set(qubits) - faulty_qubits


def get_default_ecr_schedule(backend: Backend, qubits: tuple[int, int]) -> Schedule:
    for gate in ['cx', 'ecr']:
        try:
            return backend.defaults().instruction_schedule_map.get(gate, qubits)
        except PulseError:
            pass
    return None


def get_qutrit_freq_shift(
    qubit: int,
    target: Target,
    calibrations: Calibrations
) -> float:
    f12 = calibrations.get_parameter_value('f12', qubit)
    f01 = target.qubit_properties[qubit].frequency
    return f12 - f01
