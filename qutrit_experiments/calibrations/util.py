"""Common utility functions for calibration definitions."""

from qiskit.providers import Backend
from qiskit.pulse import PulseError, Schedule


def get_operational_qubits(backend: Backend) -> set[int]:
    try:
        faulty_qubits = set(backend.properties().faulty_qubits())
    except AttributeError:
        # Simulator
        faulty_qubits = set()
    return set(range(backend.num_qubits)) - faulty_qubits


def get_default_ecr_schedule(backend: Backend, qubits: tuple[int, int]) -> Schedule:
    for gate in ['cx', 'ecr']:
        try:
            return backend.defaults().instruction_schedule_map.get(gate, qubits)
        except PulseError:
            pass
    return None
