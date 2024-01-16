from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, SXGate, XGate, UGate, U3Gate

from ..gates import RZ12Gate, SX12Gate, X12Gate, U12Gate

class EFCasted:
    def circuits(self) -> list[QuantumCircuit]:
        circuits = super().circuits()
        for circ in circuits:
            for inst in circ.data:
                op = inst.operation
                if isinstance(op, RZGate):
                    inst.operation = RZ12Gate(op.params[0])
                elif isinstance(op, SXGate):
                    inst.operation = SX12Gate()
                elif isinstance(op, XGate):
                    inst.operation = X12Gate()
                elif isinstance(op, (UGate, U3Gate)):
                    inst.operation = U12Gate(*op.params)
        return circuits


class DecomposedEFCasted:
    def circuits(self) -> list[QuantumCircuit]:
        circuits = [c.decompose() for c in super().circuits()]
        for circ in circuits:
            for inst in circ.data:
                op = inst.operation
                if isinstance(op, RZGate):
                    inst.operation = RZ12Gate(op.params[0])
                elif isinstance(op, SXGate):
                    inst.operation = SX12Gate()
                elif isinstance(op, XGate):
                    inst.operation = X12Gate()
                elif isinstance(op, (UGate, U3Gate)):
                    inst.operation = U12Gate(*op.params)
        return circuits