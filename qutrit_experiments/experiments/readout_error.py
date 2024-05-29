"""Readout confusion matrix measurements."""
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit.result import Counts
from qiskit_experiments.framework import Options
from qiskit_experiments.library import CorrelatedReadoutError as CorrelatedReadoutErrorOrig

from ..constants import DEFAULT_SHOTS
from ..transpilation import map_to_physical_qubits


class CorrelatedReadoutError(CorrelatedReadoutErrorOrig):
    """Override of CorrelatedReadoutError with custom transpilation."""
    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.shots = 10000
        return options

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = self.circuits()
        first_circuit = map_to_physical_qubits(circuits[0], self.physical_qubits,
                                               self._backend.coupling_map)

        transpiled_circuits = [first_circuit]
        # first_circuit is for 0
        for circuit in circuits[1:]:
            tcirc = first_circuit.copy()
            state_label = circuit.metadata['state_label']
            for i, val in enumerate(reversed(state_label)):
                if val == "1":
                    tcirc.data.insert(0, CircuitInstruction(XGate(), [self.physical_qubits[i]]))

            tcirc.metadata = circuit.metadata
            transpiled_circuits.append(tcirc)

        return transpiled_circuits
