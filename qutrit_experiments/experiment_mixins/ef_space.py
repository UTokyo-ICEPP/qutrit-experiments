"""A mixin to BaseExperiment to upcast all qubit gates to EF gates and add X gates to the
beginning and end of the circuits."""
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, Measure, Gate
from qiskit.circuit.library import RZGate, SXGate, XGate, UGate, U3Gate
from qiskit_experiments.framework import Options

from ..gates import QutritGate, RZ12Gate, SX12Gate, X12Gate, U12Gate


class EFSpaceExperiment:
    """A mixin to BaseExperiment to upcast all qubit gates to EF gates and add X gates to the
    beginning and end of the circuits."""
    _initial_xgate = True
    _decompose_before_cast = False

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options() # pylint: disable=no-member
        options.discrimination_basis = '01'
        return options

    def circuits(self) -> list[QuantumCircuit]:
        """Prepend and append X gates to all measured qubits."""
        circuits = []
        for circuit in super().circuits(): # pylint: disable=no-member
            if self._decompose_before_cast:
                circuit = circuit.decompose()

            for inst in circuit.data:
                op = inst.operation
                if isinstance(op, RZGate):
                    inst.operation = RZ12Gate(op.params[0])
                elif isinstance(op, SXGate):
                    inst.operation = SX12Gate()
                elif isinstance(op, XGate):
                    inst.operation = X12Gate()
                elif isinstance(op, (UGate, U3Gate)):
                    inst.operation = U12Gate(*op.params)
                elif type(op) is Gate and op.num_qubits == 1: # pylint: disable=unidiomatic-typecheck
                    inst.operation = QutritGate(name=op.name, num_qubits=1, params=list(op.params))

            qubits = (circuit.qregs[0][0],)

            if self._initial_xgate:
                circuit.data.insert(0, CircuitInstruction(XGate(), qubits))

            if self.experiment_options.discrimination_basis != '12': # pylint: disable=no-member
                idx = 0
                while idx < len(circuit.data):
                    if isinstance(circuit.data[idx].operation, Measure):
                        # 02 or 01
                        circuit.data.insert(idx, CircuitInstruction(XGate(), qubits))
                        idx += 1
                        if self.experiment_options.discrimination_basis == '01': # pylint: disable=no-member
                            circuit.data.insert(idx, CircuitInstruction(X12Gate(), qubits))
                            idx += 1
                    idx += 1

            circuits.append(circuit)

        return circuits
