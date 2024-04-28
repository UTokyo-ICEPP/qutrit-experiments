"""Transpiler passes to convert the Qutrit Toffoli gate into a fully scheduled circuit.

All passes in this module are meant to be run on a circuit that contains one Toffoli gate and
nothing else.
"""
import logging
from typing import Optional
import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Barrier, Delay, Gate
from qiskit.circuit.library import ECRGate, HGate, RZGate, SXGate, XGate
from qiskit.transpiler import (AnalysisPass, InstructionDurations, Target, TransformationPass,
                               TranspilerError)

from ..gates import (P2Gate, QutritCCXGate, QutritCCZGate, QutritQubitCXType, QutritQubitCXGate,
                     QutritQubitCZGate, QutritMCGate, RZ12Gate, XplusGate, XminusGate, X12Gate)
from .util import insert_dd

logger = logging.getLogger(__name__)


class ContainsQutritMCGate(AnalysisPass):
    """Search the DAG circuit for qutrit Toffoli gates."""
    def run(self, dag: DAGCircuit):
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritMCGate):
                self.property_set['has_qutrit_mcgate'] = True
                return


class QutritMCGateDecomposition(TransformationPass):
    """Decompose CCX or CCZ.

    The full decomposition with reverse CZ is
                                                       ░ ┌───────:::─┐┌─++++++++─┐ ░             »
    q_0: ──────────────────────────────────────────────░─┤0          ├┤ Rz(-π/2) ├─░─────────────»
         ┌─────┐┌───┐┌───────────────────┐┌─────┐┌─~─┐ ░ │  Rzx(π/2) │├─~+~+~+~+-┤ ░ ┌─────┐┌─=─┐»
    q_1: ┤ X12 ├┤ X ├┤ Delay(delay1[dt]) ├┤ X12 ├┤ X ├─░─┤1          ├┤ Rx(-π/2) ├─░─┤ X12 ├┤ X ├»
         └─────┘└───┘└───────────────────┘└─────┘└─~─┘ ░ └───────:::─┘└─~~~~~~~~-┘ ░ └─────┘└─=─┘»
    q_2: ──────────────────────────────────────────────░───────────────────────────░─────────────»
                                                       ░                           ░             »
    «                                                                                       ░ »
    «q_0: ──────────────────────────────────────────────────────────────────────────────────░─»
    «     ┌───────────┐┌───────────┐┌─======─┐┌─────┐┌───┐┌───────────────────┐┌─────┐┌─#─┐ ░ »
    «q_1: ┤1          ├┤1          ├┤ Rx(-π) ├┤ X12 ├┤ X ├┤ Delay(delay2[dt]) ├┤ X12 ├┤ X ├─░─»
    «     │  Rzx(π/2) ││  Rzx(π/2) │├─======┬┘└─────┘└───┘└───────────────────┘└─────┘└─#─┘ ░ »
    «q_2: ┤0          ├┤0          ├┤ Rz(π) ├───────────────────────────────────────────────░─»
    «     └───────────┘└───────────┘└───────┘                                               ░ »
    «     ┌─+++++++─┐┌──:::───────┐ ░
    «q_0: ┤ Rz(π/2) ├┤0           ├─░─────────────
    «     ├─#+#+#+#─┤│  Rzx(-π/2) │ ░ ┌─────┐┌───┐
    «q_1: ┤ Rx(π/2) ├┤1           ├─░─┤ X12 ├┤ X ├
    «     └─#######─┘└──:::───────┘ ░ └─────┘└───┘
    «q_2: ──────────────────────────░─────────────
    «                               ░
    The barriers are just for guiding the eye except for the first one.

    Gates with matching symbols can be cancelled / merged as:
        ┌───────:::─┐ ┌──:::───────┐   ┌──────┐┌─:─┐ ┌─:─┐┌──────┐
        ┤0          ├ ┤0           ├   ┤0     ├┤ X ├ ┤ X ├┤0     ├
        │  Rzx(π/2) │ │  Rzx(-π/2) │ = │  ECR │└─:─┘ └─:─┘│  ECR │
        ┤1          ├ ┤1           ├   ┤1     ├───── ─────┤1     ├
        └───────:::─┘ └──:::───────┘   └──────┘           └──────┘
        ┌─~─┐┌─~~~~~~~~─┐   ┌─────────┐┌──────────┐   ┌────┐┌──────────┐
        ┤ X ├┤ Rx(-π/2) ├ = ┤ Rx(π/2) ├┤ P2(-π/2) ├ = ┤ SX ├┤ P2(-π/4) ├
        └─~─┘└─~~~~~~~~─┘   └─────────┘└──────────┘   └────┘└──────────┘
        ┌─=─┐┌─======─┐   ┌──────────┐
        ┤ X ├┤ Rx(-π) ├ = ┤ P2(-π/2) ├
        └─=─┘└─======─┘   └──────────┘
        ┌─#─┐┌─#######─┐   ┌──────────┐┌──────────┐   ┌───────┐┌────┐┌────────┐┌──────────┐
        ┤ X ├┤ Rx(π/2) ├ = ┤ Rx(3π/2) ├┤ P2(-π/2) ├ = ┤ Rz(π) ├┤ SX ├┤ Rz(-π) ├┤ P2(3π/4) ├
        └─#─┘└─#######─┘   └──────────┘└──────────┘   └───────┘└────┘└────────┘└──────────┘

    The full decomposition for CRCR type X is
                                                       ░ ┌───────:::─┐┌─++++++++─┐ ░              »
    q_0: ──────────────────────────────────────────────░─┤0          ├┤ Rz(-π/2) ├─░──────────────»
         ┌─────┐┌───┐┌───────────────────┐┌─────┐┌───┐ ░ │  Rzx(π/2) │├─~+~+~+~+─┤ ░    ┌─────┐   »
    q_1: ┤ X12 ├┤ X ├┤ Delay(delay1[dt]) ├┤ X12 ├┤ X ├─░─┤1          ├┤ Rx(-π/2) ├─░────┤ X12 ├───»
         └─────┘└───┘└───────────────────┘└─────┘└───┘ ░ └───────:::─┘└─~~~~~~~~─┘ ░ ┌──┴─────┴──┐»
    q_2: ──────────────────────────────────────────────░───────────────────────────░─┤ Rx(theta) ├»
                                                       ░                           ░ └───────────┘»
    «                                                                                               ░ »
    «q_0: ──────────────────────────────────────────────────────────────────────────────────────────░─»
    «     ┌──────┐┌───┐┌──────┐┌─────┐┌──────┐┌───┐┌──────┐┌─────┐┌──────┐┌───┐┌──────┐┌──────────┐ ░ »
    «q_1: ┤0     ├┤ X ├┤0     ├┤ X12 ├┤0     ├┤ X ├┤0     ├┤ X12 ├┤0     ├┤ X ├┤0     ├┤ Rz(±π/2) ├─░─»
    «     │  CR- │└───┘│  CR- │└─────┘│  CR+ │└───┘│  CR+ │└─────┘│  CR+ │└───┘│  CR+ │└──────────┘ ░ »
    «q_2: ┤1     ├─────┤1     ├───────┤1     ├─────┤1     ├───────┤1     ├─────┤1     ├─────────────░─»
    «     └──────┘     └──────┘       └──────┘     └──────┘       └──────┘     └──────┘             ░ »
    «     ┌─+++++++─┐┌──:::───────┐ ░
    «q_0: ┤ Rz(π/2) ├┤0           ├─░─────────────
    «     ├─+++++++─┤│  Rzx(-π/2) │ ░ ┌─────┐┌───┐
    «q_1: ┤ Rx(π/2) ├┤1           ├─░─┤ X12 ├┤ X ├
    «     └─────────┘└──:::───────┘ ░ └─────┘└───┘
    «q_2: ──────────────────────────░─────────────
    «                               ░
    Most of the cancellations happen similarly. Same for CRCR type X12.
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        rcr_types: dict[tuple[int, int], int],
        apply_dd: bool = True,
        pulse_alignment: Optional[int] = None
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.rcr_types = rcr_types
        self.apply_dd = apply_dd
        self.pulse_alignment = pulse_alignment
        if self.apply_dd and not self.pulse_alignment:
            raise TranspilerError('Pulse alignment needed when applying DD')

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        cc_nodes = dag.named_nodes('qutrit_toffoli') + dag.named_nodes('qutrit_ccz')

        for node in cc_nodes:
            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.rcr_types[qids[1:]]
            is_reverse = rcr_type == QutritQubitCXType.REVERSE
            is_ccx = isinstance(node.op, QutritCCXGate)

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)

            def add_op(op, *qubits):
                subdag.apply_operation_back(op, [qreg[iq] for iq in qubits])

            if is_reverse == is_ccx:
                add_op(dur('sx', 2), 0)
                add_op(dur('sx', 2), 1)
                add_op(HGate(), 2)

            add_op(XplusGate(label='qutrit_toffoli_begin'), 1)
            add_op(X12Gate(), 1)
            add_op(SXGate(), 1)
            add_op(P2Gate(-np.pi / 4.), 1)
            add_op(Delay(dur('xplus', 1) + dur('x12', 1) + dur('sx', 1)), 0)
            add_op(Delay(dur('xplus', 1) + dur('x12', 1) + dur('sx', 1)), 2)
            add_op(Barrier(3), 0, 1, 2)
            add_op(ECRGate(), 0, 1)

            if self.apply_dd:
                cr_duration = (dur('ecr', 0, 1) - dur('x', 0)) // 2
                add_op(Delay(cr_duration), 2)
                add_op(XGate(), 2) # with X in ECR
                add_op(Delay(cr_duration), 2)
                add_op(XGate(), 2) # with X12 below

            if is_reverse:
                add_op(X12Gate(), 1)
                add_op(ECRGate(), 2, 1)
                add_op(XGate(), 2)
                add_op(ECRGate(), 2, 1)
                add_op(XGate(), 2) # coincides with X12
                add_op(P2Gate(-np.pi / 2.), 1)
                add_op(RZGate(-np.pi), 2)
                add_op(XplusGate(), 1)
                interval = dur('x12', 1) + 2 * dur('ecr', 2, 1) + dur('x', 2)
                add_op(Delay(interval - dur('xplus', 1)), 1)
                add_op(X12Gate(), 1)
                add_op(RZGate(np.pi), 1)
                add_op(SXGate(), 1)
                add_op(RZGate(-np.pi), 1)
                add_op(P2Gate(3. * np.pi / 4.), 1)

                if self.apply_dd:
                    cr_duration = (dur('ecr', 2, 1) - dur('x', 2)) // 2
                    add_op(XGate(), 0) # with first X12
                    add_op(Delay(cr_duration), 0)
                    add_op(XGate(), 0) # with X in ECR
                    add_op(Delay(cr_duration), 0)
                    add_op(XGate(), 0) # with X between ECRs
                    add_op(Delay(cr_duration), 0)
                    add_op(XGate(), 0) # with X in ECR
                    add_op(Delay(cr_duration), 0)
                    dd_unit = (interval + dur('x12', 1) + dur('sx', 1) - 2 * dur('x', 0)) // 4
                    add_op(Delay(dd_unit), 0)
                    add_op(XGate(), 0)
                    add_op(Delay(2 * dd_unit), 0)
                    add_op(XGate(), 0) # with SX
                    add_op(Delay(dd_unit), 0)

                    add_op(Delay(dd_unit), 2)
                    add_op(XGate(), 2)
                    add_op(Delay(2 * dd_unit - dur('x', 2)), 2)
                    add_op(XGate(), 2)
                    add_op(Delay(dd_unit), 2)
            else:
                add_op(QutritQubitCXGate(), 1, 2)
                add_op(SXGate(), 1)
                add_op(P2Gate(np.pi / 4.), 1)

                if self.apply_dd:
                    dd_delay = dur(f'qutrit_qubit_cx_rcr{rcr_type}', 1, 2) // 6 - dur('x', 0)
                    if rcr_type == QutritQubitCXType.X:
                        add_op(XGate(), 0)
                    for _ in range(5):
                        add_op(Delay(dd_delay), 0)
                        add_op(XGate(), 0)
                    add_op(Delay(dd_delay), 0)
                    if rcr_type == QutritQubitCXType.X12:
                        add_op(XGate(), 0)

            add_op(ECRGate(), 0, 1)

            if self.apply_dd:
                cr_duration = (dur('ecr', 0, 1) - dur('x', 0)) // 2
                add_op(Delay(cr_duration), 2)
                add_op(XGate(), 2)
                add_op(Delay(cr_duration), 2)
                add_op(XGate(), 2) # with the last X12
                add_op(Delay(dur('x', 2)), 2)

            add_op(XplusGate(label='qutrit_toffoli_end'), 1)
            add_op(Delay(dur('xplus', 1)), 0)
            if is_reverse == is_ccx:
                add_op(Delay(dur('sx', 2)), 0)
                add_op(Delay(dur('sx', 2)), 1)
                add_op(HGate(), 2)

            dag.substitute_node_with_dag(node, subdag)

        return dag


class QutritToffoliRefocusing(TransformationPass):
    """Calculate the phase errors due to f12 detuning and convert the last X+ to X-X- with an
    inserted delay.

    With f12 detuning, X12->X12 exp(-iδtζ). Effect on CX(c2, t) is a phase shift on the |2> level.
    The general structure of the Toffoli gate is
        T = X+(t2) U P2(δs) X+(t1) D(τ) X+(t0)
    Since X+(t) = X+ exp(-iδtζ),
        T = X+ U X+ D(τ) X+ P0[δ(t2 - t1 + s)] P1[δ(t1 - t0)] P2[δ(-t2 + t0)].
    We want the P0 and P1 arguments to coincide so that T has zero phase error in the qubit space.
    Let
        t1 - t0 = τ + a (a = X+ duration)
        t2 - t1 = d
    then
        d + s = τ + a
        ∴ τ = d + s - a

    The values of s are given by the following:
    Type CRCR_X
        CR+ X CR+ X12(t2) CR+ X CR+ X12(t1) CR- X CR- X12(t0)
      = CR+ X CR+ X12 CR+ X CR+ X12 CR- X CR- X12 exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CX exp[iδα(z+2ζ)] ~ CX P2(-3α)
    where α is the time between the two X12s = 2*CR+X+X12
    Type CRCR_X12
        X CR- X12(t2) CR- X CR+ X12(t1) CR+ X CR+ X12(t0) CR+
      = X CR- X12 CR- X CR+ X12 CR+ X CR+ X12 CR+ exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CX exp[iδα(z+2ζ)] ~ CX P2(-3α)
    Type REVERSE
        X+(t2)I Delay X+(t1)I ECR IX ECR X12(t0)X
      = X+I Delay X+I ECR IX ECR X12 exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CZ exp[iδα(z+2ζ)] ~ CZ P2(-3α)
    where α is the time between the first two X12s = 2*(ECR+IX)

    It could happen that d + s - a < 0. In such cases, we replace X+(t1) D(τ) X+(t0) with X-(t1).
    Here t1 refers to the timing of the X12 pulse, which does not change going from X+ to X- because
    the X of the X+ in the former case is cancelled. Phase error is then
        T = X+ U X- P0(δs) exp[iδ(t2 - t1)(z + ζ)].
    To cancel the phase on |0> we need
        t2 - t1 = d = -s.
    If 0 < d + s < a, we can be achieved the above condition by adjusting some delays. For all three
    cases above, s = -3α where α is
    the duration of a sequence that is repeated twice. By adding a delay Δ to this sequence,
        d -> d' = d + 2Δ
        s -> s' = s - 3Δ
        ∴ d' + s' = d + s - Δ
    We are out of luck if d + s < 0.

    Since we introduce new gates and delays, node_start_time will be invalidated in this pass.
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        rcr_types: dict[tuple[int, int], QutritQubitCXType],
        apply_dd: bool,
        pulse_alignment: int
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.rcr_types = rcr_types
        self.apply_dd = apply_dd
        self.pulse_alignment = pulse_alignment

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Invalidate node_start_time
        node_start_time = self.property_set.pop('node_start_time')

        toffoli_begins = [node for node in dag.topological_op_nodes()
                          if node.op.label == 'qutrit_toffoli_begin']

        for begin_node in toffoli_begins:
            c2 = begin_node.qargs[0]
            x12_node = next(dag.successors(begin_node))
            node = x12_node
            while True:
                if len((node := next(dag.successors(node))).qargs) == 3:
                    break
            barrier = node
            c1 = barrier.qargs[0]
            t = barrier.qargs[2]

            qids = tuple(dag.find_bit(q).index for q in [c1, c2, t])

            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            def add_op(op, *iqs):
                subdag.apply_operation_back(op, [qreg[iq] for iq in iqs])

            if any((isinstance(node, DAGOpNode) and isinstance(node.op, HGate))
                   for node in dag.ancestors(barrier)):
                add_op(dur('sx', 2), 0)
                add_op(dur('sx', 2), 1)
                add_op(HGate(), 2)

            if self.rcr_types[qids[1:]] == QutritQubitCXType.REVERSE:
                alpha = dur('x12', 1) + 2 * dur('ecr', 2, 1) + dur('x', 2)
            else:
                alpha = 2 * dur('cr', 1, 2) + dur('x', 1) + dur('x12', 1)

            node = barrier
            while True:
                node = next(n for n in dag.successors(node)
                            if isinstance(n, DAGOpNode) and c2 in n.qargs)
                if isinstance(node.op, XplusGate) and node.op.label == 'qutrit_toffoli_end':
                    end_node = node
                    break
            # d + s
            # For the reverse CX implementation, delay >= 0 only if T[ECR01] >= T[ECR21]. If
            # otherwise, we need to adjust the timing in the CZ sequence
            added_time = -3 * alpha + (node_start_time[end_node] - node_start_time[x12_node])

            if added_time >= dur('xplus', 1):
                add_op(XplusGate(label='qutrit_toffoli_begin'), 1)
                add_op(Delay(added_time - dur('xplus', 1)), 1)
                add_op(X12Gate(), 1)
                add_op(SXGate(), 1)
                add_op(P2Gate(-np.pi / 4.), 1)
                if self.apply_dd:
                    total_delay = added_time + dur('x12', 1) + dur('sx', 1) - 2 * dur('x', 0)
                    dd_unit = total_delay // 2
                    add_op(Delay(dd_unit), 0)
                    add_op(XGate(), 0)
                    add_op(Delay(dd_unit), 0)
                    add_op(XGate(), 0)
                    add_op(Delay(dd_unit), 2)
                    add_op(XGate(), 2)
                    add_op(Delay(dd_unit), 2)
                    add_op(XGate(), 2)
            else:
                add_op(XminusGate(label='qutrit_toffoli_begin'), 1)
                add_op(RZGate(np.pi), 1)
                add_op(SXGate(), 1)
                add_op(RZGate(-np.pi), 1)
                add_op(P2Gate(np.pi / 4.), 1)
                raise NotImplementedError('Someday I will')

            # dag.remove_ancestors_of(barrier._node_id) # Does not work due to a bug
            for anc in dag.ancestors(barrier):
                if isinstance(anc, DAGOpNode):
                    dag.remove_op_node(anc)
            dag.substitute_node_with_dag(barrier, subdag)

        return dag
