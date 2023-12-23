from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Measure
from qiskit.circuit.library import XGate
from qiskit_experiments.framework import Options

from ..gates import X12Gate


class EFSpaceExperiment:
    """A mixin to BaseExperiment to add X gates to the beginning and end of the circuits."""
    _initial_xgate = True

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.final_xgate = True
        options.final_x12gate = True
        return options

    def circuits(self) -> list[QuantumCircuit]:
        """Prepend and append X gates to all measured qubits."""
        circuits = []
        for circuit in super().circuits():
            new_circuit = circuit.copy_empty_like()

            if self._initial_xgate:
                new_circuit.x(0)

            new_circuit.compose(circuit, inplace=True)

            if self.experiment_options.final_xgate:
                idx = 0
                while idx < len(new_circuit.data):
                    if isinstance(new_circuit.data[idx].operation, Measure):
                        qubits = new_circuit.data[idx].qubits
                        new_circuit.data.insert(idx, CircuitInstruction(XGate(), qubits))
                        idx += 1
                        if self.experiment_options.final_x12gate:
                            new_circuit.data.insert(idx, CircuitInstruction(X12Gate(), qubits))
                            idx += 1
                    idx += 1

            circuits.append(new_circuit)

        return circuits
