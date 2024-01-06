# pylint: disable=no-member
"""Mixin for trivial transpilation."""
from qiskit import QuantumCircuit

from ..transpilation import map_to_physical_qubits


class MapToPhysicalQubits:
    """Mixin for trivial transpilation."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                      self._backend.coupling_map)


class MapToPhysicalQubitsCal:
    """Trivial transpilation mixin for calibration experiments."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        transpiled = map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                            self._backend.coupling_map)
        for circ in transpiled:
            self._attach_calibrations(circ)

        return transpiled