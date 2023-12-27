# pylint: disable=no-member
"""Mixin for trivial transpilation."""
from qiskit import QuantumCircuit

from ..transpilation import map_to_physical_qubits


class MapToPhysicalQubits:
    """Mixin for trivial transpilation."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                      self._backend.coupling_map)
