from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Measure
from qiskit.circuit.library import XGate


class EFSpaceExperiment:
    """A mixin to BaseExperiment to add X gates to the beginning and end of the circuits."""
    _initial_xgate = True
    _final_xgate = True

    def circuits(self) -> list[QuantumCircuit]:
        """Prepend and append X gates to all measured qubits."""
        circuits = []
        for circuit in super().circuits():
            new_circuit = circuit.copy_empty_like()

            if self._initial_xgate:
                new_circuit.x(0)

            new_circuit.compose(circuit, inplace=True)

            if self._final_xgate:
                idx = 0
                while idx < len(new_circuit.data):
                    if isinstance(new_circuit.data[idx].operation, Measure):
                        xinst = CircuitInstruction(XGate(), (new_circuit.qregs[0][0],))
                        new_circuit.data.insert(idx, xinst)
                        idx += 2
                    else:
                        idx += 1

            circuits.append(new_circuit)

        return circuits
