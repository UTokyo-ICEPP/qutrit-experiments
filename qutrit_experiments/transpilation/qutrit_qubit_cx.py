import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import ECRGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionDurations, Target, TransformationPass

from ..gates import QutritQubitCXGate, QutritQubitCXType, RZ12Gate, X12Gate, XplusGate
from .util import insert_dd


class ReverseCXDecomposition(TransformationPass):
    """Decompose QutritQubitCXGate to the double reverse ECR sequence with basis cycling."""
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        rcr_types: dict[tuple[int, int], int]
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.rcr_types = dict(rcr_types)
        self.dummy_circuit = False

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, QutritQubitCXGate):
                continue

            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            if self.rcr_types[qids] != QutritQubitCXType.REVERSE:
                  continue

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            subdag = DAGCircuit()
            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)
            t = 0
            subdag.apply_operation_back(XplusGate(), [qreg[0]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(SXGate(), [qreg[1]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
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
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(SXGate(), [qreg[1]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(Barrier(2), qreg)
            subdag.apply_operation_back(XGate(), [qreg[0]])
            subdag.apply_operation_back(Delay(t - dur('xplus', 0)), [qreg[0]])
            subdag.apply_operation_back(XplusGate(label='reverse_cx_end'), [qreg[0]])

            dag.substitute_node_with_dag(node, subdag)

        return dag


class ReverseCXDynamicalDecoupling(TransformationPass):
    def __init__(self, target: Target, instruction_durations: InstructionDurations):
        super().__init__()
        self.target = target
        self.inst_durations = instruction_durations

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for last_xplus in [node for node in dag.named_nodes('xplus')
                           if node.op.label == 'reverse_cx_end']:
            delay_node = next(dag.predecessors(last_xplus))
            x_node = next(dag.predecessors(delay_node))
            barrier_node = next(dag.predecessors(x_node))
            physical_qubits = tuple(dag.find_bit(q).index for q in barrier_node.qargs)

            x_duration = self.inst_durations.get('x', [physical_qubits[1]])
            xplus_duration = self.inst_durations.get('xplus', [physical_qubits[0]])

            subdag = DAGCircuit()
            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)

            subdag.apply_operation_back(Barrier(2), qreg)
            for duration in [x_duration + delay_node.op.params[0], xplus_duration]:
                insert_dd(subdag, qreg[1], duration, x_duration, self.target.pulse_alignment)

            dag.substitute_node_with_dag(barrier_node, subdag)

        return dag
