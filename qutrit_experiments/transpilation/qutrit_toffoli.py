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
                     QutritQubitCZGate, QutritMCGate, RZ12Gate, XplusGate, X12Gate)
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
                add_op(HGate(), 2)
                add_op(Barrier(3), 0, 1, 2)                

            add_op(XplusGate(label='qutrit_toffoli_begin'), 1)
            add_op(Delay(0), 1)
            add_op(X12Gate(), 1)

            add_op(SXGate(), 1)
            add_op(P2Gate(-np.pi / 4.), 1)

            if self.apply_dd:
                add_op(Delay(0), 0)
                add_op(XGate(), 0)
                add_op(Delay(0), 0)
                add_op(XGate(), 0)
                add_op(Delay(0), 2)
                add_op(XGate(), 2)
                add_op(Delay(0), 2)
                add_op(XGate(), 2) # coincides with SX

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
                add_op(P2Gate(-np.pi / 2.), 1)
                add_op(ECRGate(), 2, 1)
                add_op(XGate(), 2)
                add_op(ECRGate(), 2, 1)
                add_op(XGate(), 2) # coincides with X12
                add_op(RZGate(np.pi), 2)
                add_op(XplusGate(), 1)
                interval = dur('x12', 1) + 2 * dur('ecr', 2, 1) + dur('x', 2)
                delay = interval - dur('xplus', 1)
                add_op(Delay(delay), 1)
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
                    add_op(Delay(interval // 2), 0)
                    add_op(XGate(), 0)
                    add_op(Delay(interval // 2), 0)
                    add_op(XGate(), 0) # with SX

                    add_op(Delay(interval // 2), 2)
                    add_op(XGate(), 2)
                    add_op(Delay(interval // 2), 2)
                    add_op(XGate(), 2) # with the beginning of ECR
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

            add_op(XplusGate(label='qutrit_toffoli_end'), 1)
            if (self.rcr_types[qids[1:]] == QutritQubitCXType.REVERSE
                and isinstance(node.op, QutritCCXGate)):
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

    Since we introduce new gates and delays, node_start_time will be invalidated after this pass.
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        rcr_types: dict[tuple[int, int], QutritQubitCXType],
        pulse_alignment: int
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.rcr_types = rcr_types
        self.pulse_alignment = pulse_alignment

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Invalidate node_start_time
        node_start_time = self.property_set.pop('node_start_time')

        toffoli_begins = [node for node in dag.topological_op_nodes()
                          if node.op.label == 'qutrit_toffoli_begin']

        for begin_node in toffoli_begins:
            c2 = begin_node.qargs[0]
            delay_node = next(dag.successors(begin_node))
            x12_node = next(dag.successors(delay_node)) # X12 at t1
            node = x12_node
            while True:
                if len((node := next(dag.successors(node))).qargs) == 3:
                    break
            barrier = node
            c1 = barrier.qargs[0]
            t = barrier.qargs[2]

            qids = tuple(dag.find_bit(q).index for q in [c1, c2, t])

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            c1_dd_delays = []
            t_dd_delays = []
            try:
                c1_x_node = next(node for node in dag.predecessors(barrier)
                                 if isinstance(node, DAGOpNode) and node.qargs == (c1,))
            except StopIteration:
                pass
            else:
                # Using DD
                c1_dd_delays = [(node := next(dag.predecessors(c1_x_node)))]
                node = next(dag.predecessors(node))
                c1_dd_delays.append(next(dag.predecessors(node)))
                t_x_node = next(node for node in dag.predecessors(barrier) if node.qargs == (t,))
                t_dd_delays = [(node := next(dag.predecessors(t_x_node)))]
                node = next(dag.predecessors(node))
                t_dd_delays.append(next(dag.predecessors(node)))

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
            delay_node.op = delay_node.op.to_mutable()
            delay_node.op.duration = added_time - dur('xplus', 1) # d + s - a
            for iq, delays_list in zip([0, 2], [c1_dd_delays, t_dd_delays]):
                dd_delay = (added_time + dur('xplus', 1) - 2 * dur('x', iq)) // 2
                dd_delay //= self.pulse_alignment
                dd_delay *= self.pulse_alignment
                for node in delays_list:
                    node.op = node.op.to_mutable()
                    node.op.duration = dd_delay

        return dag


# class QutritToffoliDynamicalDecoupling(TransformationPass):
#     """Insert DD sequences to idle qubits. More aggressive than PadDynamicalDecoupling."""
#     def __init__(
#         self,
#         target: Target,
#         instruction_durations: InstructionDurations,
#         rcr_types: int
#     ):
#         super().__init__()
#         self.target = target
#         self.inst_durations = instruction_durations
#         self.rcr_types = rcr_types

#     def run(self, dag: DAGCircuit) -> DAGCircuit:
#         toffoli_begins = [node for node in dag.topological_op_nodes()
#                           if node.op.label == 'qutrit_toffoli_begin']

#         for begin_node in toffoli_begins:

#         def make_dd_subdag():
#             subdag = DAGCircuit()
#             qreg = QuantumRegister(3)
#             subdag.add_qreg(qreg)
#             return subdag, qreg, []

#         def dur(gate, *iqs):
#             return self.inst_durations.get(gate, [qids[i] for i in iqs])

#         def add_dd(iq, duration, placement='right'):
#             delay = (duration - 2 * dur('x', iq)) // 2
#             if placement == 'right':
#                 subdag.apply_operation_back(Delay(delay), [qreg[iq]])
#             subdag.apply_operation_back(XGate(), [qreg[iq]])
#             subdag.apply_operation_back(Delay(delay), [qreg[iq]])
#             subdag.apply_operation_back(XGate(), [qreg[iq]])
#             if placement == 'left':
#                 subdag.apply_operation_back(Delay(delay), [qreg[iq]])

#         node_start_time = self.property_set["node_start_time"]
#         ton = dag.topological_op_nodes()

#         first_xplus_node = next(node for node in ton if isinstance(node.op, XplusGate))
#         c2_qubit = first_xplus_node.qargs[0]
#         first_ecr_node = next(node for node in ton
#                              if isinstance(node.op, Gate) and len(node.qargs) == 2)
#         first_rz_node = next(node for node in dag.predecessors(first_ecr_node)
#                              if isinstance(node.op, RZGate))
#         c1_qubit = next(qarg for qarg in first_ecr_node.qargs if qarg != c2_qubit)
#         t_qubit = next(node.qargs[0] for node in ton
#                        if isinstance(node.op, Gate) and node.qargs[0] not in (c1_qubit, c2_qubit))
#         refocusing_delay = next(node.op.duration for node in ton if isinstance(node.op, Delay))

#         qids = tuple(dag.find_bit(q).index for q in [c1_qubit, c2_qubit, t_qubit])

#         # Insert c1 DD during the first xplus-delay-xplus
#         subdag, qreg, _ = make_dd_subdag()
#         duration = 2 * (dur('x12', 1) + dur('x', 1)) + refocusing_delay
#         add_dd(0, duration)
#         subdag.apply_operation_back(first_rz_node.op, [qreg[0]])
#         dag.substitute_node_with_dag(first_rz_node, subdag)

#         first_t_node = next(node for node in dag.topological_nodes()
#                             if isinstance(node.op, Gate) and node.qargs == (t_qubit,))

#         subdag, qreg, _ = make_dd_subdag()
#         duration = 2 * (dur('x12', 1) + dur('x', 1)) + refocusing_delay
#         add_dd(2, duration)
#         duration = dur('ecr', 0, 1) + dur('x', 0)
#         add_dd(2, duration)
#         subdag.apply_operation_back(first_t_node.op, [qreg[2]])
#         dag.substitute_node_with_dag(first_ecr_node, subdag)

#         # TODO MOVE THE FOLLOWING block to qutrit_qubit_cx
#         # Reaccept rcr_types both here and in cx
#         # Perhaps refocusing can also be done before decomposing cx - makes double-handling by rcr types
#         # easier

#         ecr_x_node = next(node for node in dag.successors(first_ecr_node)
#                           if node.qargs == (c1_qubit,))
#         subdag, qreg, _ = make_dd_subdag()
#         subdag.apply_operation_back(ecr_x_node, [qreg[0]])
#         if self.rcr_type == QutritQubitCXType.REVERSE:
#             duration = 2 * (dur('x', 2) + dur('ecr', 2, 1))
#             add_dd(0, duration, 'left')
#             add_dd(0, duration, 'left')
#             add_dd(0, 2 * duration)
#         else:
#             raise NotImplementedError('CRCR case not implemented')
#         dag.substitute_node_with_dag(ecr_x_node, subdag)

#         # Analyze the gate structure (qargs) first
#         first_barrier = next(node for node in dag.topological_op_nodes()
#                              if isinstance(node.op, Barrier))
#         circuit_qargs = first_barrier.qargs
#         barriers = [first_barrier]
#         # Get the first X+
#         node = next(s for s in dag.successors(first_barrier) if isinstance(s.op, XplusGate))
#         c2_qubit = node.qargs[0]
#         logger.debug('xplus0 on %s found at t=%d', c2_qubit, node_start_time[node])
#         xplus_times = [node_start_time[node]]
#         # Follow the graph for the next X+
#         while True:
#             if isinstance((node := next(dag.successors(node))).op, XplusGate):
#                 break
#         logger.debug('xplus1 found at t=%d', node_start_time[node])
#         xplus_times.append(node_start_time[node])
#         barriers.append(next(dag.successors(node)))
#         # Follow the graph until we hit a two-qubit gate between c1 and c2
#         while True:
#             if len((node := next(dag.successors(node))).qargs) == 2:
#                 break
#         logger.debug('CX/ECR(0, 1) on %s found at t=%d', node.qargs, node_start_time[node])
#         c1_qubit = node.qargs[0]
#         t_qubit = next(q for q in circuit_qargs if q not in [c2_qubit, c1_qubit])
#         # Follow the graph until we hit a barrier before c2-t CX
#         while True:
#             if isinstance((node := next(dag.successors(node))).op, Barrier):
#                 break
#         barriers.append(node)
#         # Follow the graph (skip the c2-t CX) until the next Barrier(3)
#         # (need qargs condition because reverse cx uses a Barrier(2) internally)
#         while True:
#             if isinstance((node := next(dag.successors(node))).op, Barrier) and len(node.qargs) == 3:
#                 break
#         barriers.append(node)
#         # Follow until the next barrier(2)
#         node = next(s for s in dag.successors(node) if t_qubit not in s.qargs)
#         while True:
#             if isinstance((node := next(dag.successors(node))).op, Barrier):
#                 break
#         barriers.append(node)
#         # Find the final barrier
#         while True:
#             if isinstance((node := next(dag.successors(node))).op, Barrier):
#                 break
#         barriers.append(node)

#         # Insert DD sequences
#         qids = tuple(dag.find_bit(q).index for q in [c1_qubit, c2_qubit, t_qubit])
#         logger.debug('qids %s', qids)
#         xplus_duration = self.inst_durations.get('xplus', [qids[1]])
#         x_durations = [self.inst_durations.get('x', [qid]) for qid in qids]
#         rcr_type = self.rcr_types[qids[1:]]

#         def make_dd_subdag():
#             subdag = DAGCircuit()
#             qreg = QuantumRegister(3)
#             subdag.add_qreg(qreg)
#             return subdag, qreg, []

#         def add_dd(qubit, start_time, duration, placement='left'):
#             times = insert_dd(subdag, qreg[qubit], duration, x_durations[qubit],
#                               self.target.pulse_alignment, start_time=start_time,
#                               placement=placement)
#             start_times.extend(times)

#         def insert_dd_to_dag(node_to_replace):
#             subst_map = dag.substitute_node_with_dag(node_to_replace, subdag)
#             for node, time in start_times:
#                 node_start_time[subst_map[node._node_id]] = time

#         # C1&T DD (begin side)
#         subdag, qreg, start_times = make_dd_subdag()

#         node = subdag.apply_operation_back(Barrier(3), qreg)
#         start_time = node_start_time.pop(barriers[0])
#         start_times.append((node, start_time))
#         for time, duration in [
#             (start_time, xplus_times[1] + xplus_duration),
# #            (node_start_time[barriers[1]] - xplus_duration, xplus_duration)
#         ]:
#             add_dd(0, time, duration)
#             add_dd(2, time, duration)
#         add_dd(2, node_start_time[barriers[1]],
#                node_start_time[barriers[2]] - node_start_time[barriers[1]], placement='right')

#         insert_dd_to_dag(barriers[0])

#         # C1&T DD (end side)
#         subdag, qreg, start_times = make_dd_subdag()

#         start_time = node_start_time[barriers[-3]]
#         add_dd(2, start_time, node_start_time[barriers[-2]] - start_time)
#         add_dd(0, node_start_time[barriers[-2]], xplus_duration)
#         add_dd(2, node_start_time[barriers[-2]], xplus_duration)
#         node = subdag.apply_operation_back(Barrier(3), qreg)
#         start_times.append((node, node_start_time.pop(barriers[-1])))

#         insert_dd_to_dag(barriers[-1])

#         # C1 DD (during qutrit-qubit CX)
#         subdag, qreg, start_times = make_dd_subdag()

#         node = subdag.apply_operation_back(Barrier(3), qreg)
#         start_time = node_start_time.pop(barriers[2])
#         start_times.append((node, start_time))

#         if rcr_type == QutritQubitCXType.REVERSE:
#             ecr_duration = self.inst_durations.get('ecr', (qids[2], qids[1]))

#             time = start_time
#             add_dd(0, time, xplus_duration)
#             time += xplus_duration
#             add_dd(0, time, ecr_duration + x_durations[2], 'right')
#             time += ecr_duration + x_durations[2]
#             add_dd(0, time, ecr_duration + x_durations[2], 'right')
#             time += ecr_duration + x_durations[2]
#             # From the X of second X+ to just before the X12 of the last X+
#             end_time = node_start_time[barriers[3]] - xplus_duration
#             add_dd(0, time, end_time - time)
#             time = end_time
#             add_dd(0, time, xplus_duration)
#         else:
#             t_dd_duration = x_durations[2] * 2
#             cr_duration = self.inst_durations.get('cr', qids[1:])
#             sx_duration = self.inst_durations.get('sx', [qids[2]])
#             time = start_time
#             if rcr_type == QutritQubitCXType.X:
#                 for _ in range(3):
#                     add_dd(0, time, t_dd_duration)
#                     time += t_dd_duration
#                     add_dd(0, time, cr_duration)
#                     time += cr_duration
#                     add_dd(0, time, t_dd_duration)
#                     time += t_dd_duration
#                     add_dd(0, time, cr_duration)
#                     time += cr_duration
#             else:
#                 for _ in range(3):
#                     add_dd(0, time, cr_duration)
#                     time += cr_duration
#                     add_dd(0, time, t_dd_duration)
#                     time += t_dd_duration
#                     add_dd(0, time, cr_duration)
#                     time += cr_duration
#                     add_dd(0, time, t_dd_duration)
#                     time += t_dd_duration
#             add_dd(0, time, 2 * sx_duration) # cx_offset_rx duration

#         insert_dd_to_dag(barriers[2])

#         return dag
