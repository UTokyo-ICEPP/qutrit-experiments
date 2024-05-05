from typing import Optional
import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import ECRGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionDurations, Target, TransformationPass, TranspilerError

from ..gates import P2Gate, QutritQubitCXGate, QutritQubitCXType, RZ12Gate, X12Gate, XplusGate
from .util import insert_dd


class ReverseCXDecomposition(TransformationPass):
    """Decompose QutritQubitCXGate to the double reverse ECR sequence with basis cycling.

    The circuit for CZ with basis cycling is the following:
         ┌─────┐┌─=─┐┌───────────┐┌───────────┐┌─=====─┐┌─────┐┌───┐┌───────────────────┐┌─────┐┌───┐
    q_0: ┤ X12 ├┤ X ├┤1          ├┤1          ├┤ Rx(-π)├┤ X12 ├┤ X ├┤ Delay(delay2[dt]) ├┤ X12 ├┤ X ├
         └─────┘└─=─┘│  Rzx(π/2) ││  Rzx(π/2) │├─=====┬┘└─────┘└───┘└───────────────────┘└─────┘└───┘
    q_1: ────────────┤0          ├┤0          ├┤ Rz(π)├──────────────────────────────────────────────
                     └───────────┘└───────────┘└──────┘
    where X and Rx(-π) cancel to P2(-π/2).
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        apply_dd: bool = True,
        pulse_alignment: Optional[int] = None
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.apply_dd = apply_dd
        self.pulse_alignment = pulse_alignment
        if self.apply_dd and not self.pulse_alignment:
            raise TranspilerError('Pulse alignment needed when applying DD')
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cx_cz_nodes = dag.named_nodes('qutrit_qubit_cx') + dag.named_nodes('qutrit_qubit_cz')
        for node in cx_cz_nodes:
            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids)
            if rcr_type != QutritQubitCXType.REVERSE:
                  continue

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            subdag = DAGCircuit()
            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)

            def add_op(op, *qubits):
                subdag.apply_operation_back(op, [qreg[iq] for iq in qubits])

            if isinstance(node.op, QutritQubitCXGate):
                add_op(Delay(dur('sx', 1)), 0)
                add_op(HGate(), 1)
            add_op(X12Gate(), 0)
            add_op(ECRGate(), 1, 0)
            add_op(XGate(), 1)
            add_op(ECRGate(), 1, 0)
            add_op(XGate(), 1)
            add_op(P2Gate(-np.pi / 2.), 0)
            add_op(RZGate(-np.pi), 1)
            add_op(XplusGate(), 0)
            interval = dur('x12', 0) + 2 * dur('ecr', 1, 0) + dur('x', 1)
            add_op(Delay(interval - dur('xplus', 0)), 0)
            add_op(XplusGate(), 0)

            if self.apply_dd:
                dd_unit = (interval + 2 * dur('x', 0) - 2 * dur('x', 1)) // 4
                add_op(Delay(dd_unit), 1)
                add_op(XGate(), 1)
                add_op(Delay(2 * dd_unit - dur('x', 1)), 1)
                add_op(XGate(), 1)
                add_op(Delay(dd_unit), 1)

            if isinstance(node.op, QutritQubitCXGate):
                add_op(Delay(dur('sx', 1)), 0)
                add_op(HGate(), 1)

            dag.substitute_node_with_dag(node, subdag)

        return dag
