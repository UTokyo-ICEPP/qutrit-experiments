import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import ECRGate, HGate, RZGate, XGate
from qiskit.transpiler import InstructionDurations, TransformationPass

from ..gates import QutritQubitCXGate, RZ12Gate, X12Gate, XplusGate


class ReverseCXDecomposition(TransformationPass):
    """Decompose QutritQubitCXGate to the double reverse ECR sequence with basis cycling."""
    def __init__(self, instruction_durations: InstructionDurations):
        super().__init__()
        self.inst_durations = instruction_durations
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, QutritQubitCXGate):
                continue

            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids)
            if rcr_type != -1:
                  continue

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            subdag = DAGCircuit()
            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)
            t = 0
            subdag.apply_operation_back(XplusGate(), [qreg[0]])
            subdag.apply_operation_back(HGate(), [qreg[1]])
            subdag.apply_operation_back(XGate(), [qreg[1]])
            t += max(dur('xplus', 0), dur('sx', 1) + dur('x', 1))
            subdag.apply_operation_back(ECRGate(), [qreg[1], qreg[0]])
            t += dur('ecr', 1, 0)
            subdag.apply_operation_back(XGate(), [qreg[0]])
            subdag.apply_operation_back(RZGate(np.pi / 3.), [qreg[0]]) # Cancel the geometric phase correction
            subdag.apply_operation_back(RZ12Gate(2. * np.pi / 3.), [qreg[0]]) # Cancel the geometric phase correction
            subdag.apply_operation_back(XGate(), [qreg[1]])
            t += max(dur('x', 0), dur('x', 1))
            subdag.apply_operation_back(ECRGate(), [qreg[1], qreg[0]])
            t += dur('ecr', 1, 0)
            subdag.apply_operation_back(X12Gate(), [qreg[0]]) # We need an X+ here but would like to insert a barrier in between
            subdag.apply_operation_back(RZGate(np.pi), [qreg[1]])
            subdag.apply_operation_back(HGate(), [qreg[1]])
            subdag.apply_operation_back(Barrier(2), qreg)
            subdag.apply_operation_back(XGate(), [qreg[0]])
            subdag.apply_operation_back(Delay(t - dur('xplus', 0)), [qreg[0]])
            subdag.apply_operation_back(XplusGate(label='reverse_cx_end'), [qreg[0]])

            dag.substitute_node_with_dag(node, subdag)

        return dag