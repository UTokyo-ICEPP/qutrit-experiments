"""Transpiler passes to convert the Qutrit Toffoli gate into a fully scheduled circuit.

All passes in this module are meant to be run on a circuit that contains one Toffoli gate and
nothing else.
"""
import logging
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Delay
from qiskit.circuit.library import HGate, RZGate, SXGate, XGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import (AnalysisPass, InstructionDurations, TransformationPass,
                               TranspilerError)
from qiskit_experiments.exceptions import CalibrationError

from ..gates import (P2Gate, QutritQubitCXType, QutritQubitCXGate, QutritMCGate, XplusGate,
                     XminusGate, X12Gate)
from .dynamical_decoupling import DDCalculator
from .qutrit_qubit_cx import reverse2q_decomposition_circuit

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
    «                                                                                   ░ »
    «q_0: ──────────────────────────────────────────────────────────────────────────────░─»
    «     ┌──────┐┌───┐┌──────┐┌─────┐┌──────┐┌───┐┌──────┐┌─────┐┌──────┐┌───┐┌──────┐ ░ »
    «q_1: ┤0     ├┤ X ├┤0     ├┤ X12 ├┤0     ├┤ X ├┤0     ├┤ X12 ├┤0     ├┤ X ├┤0     ├─░─»
    «     │  CR- │└───┘│  CR- │└─────┘│  CR+ │└───┘│  CR+ │└─────┘│  CR+ │└───┘│  CR+ │ ░ »
    «q_2: ┤1     ├─────┤1     ├───────┤1     ├─────┤1     ├───────┤1     ├─────┤1     ├─░─»
    «     └──────┘     └──────┘       └──────┘     └──────┘       └──────┘     └──────┘ ░ »
    «                 ┌─+++++++─┐┌──:::───────┐ ░
    «q_0: ────────────┤ Rz(π/2) ├┤0           ├─░─────────────
    «     ┌──────────┐├─+++++++─┤│  Rzx(-π/2) │ ░ ┌─────┐┌───┐
    «q_1: ┤ Rz(±π/2) ├┤ Rx(π/2) ├┤1           ├─░─┤ X12 ├┤ X ├
    «     └──────────┘└─────────┘└──:::───────┘ ░ └─────┘└───┘
    «q_2: ──────────────────────────────────────░─────────────
    «                                           ░
    Most of the cancellations happen similarly. Same for CRCR type X12.
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
        cc_nodes = dag.named_nodes('qutrit_toffoli') + dag.named_nodes('qutrit_ccz')

        for node in cc_nodes:
            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids[1:])
            try:
                delta_cz = self.calibrations.get_parameter_value('delta_cz', qids)
            except CalibrationError:
                delta_cz = 0.
            try:
                delta_ccz = self.calibrations.get_parameter_value('delta_ccz', qids)
            except CalibrationError:
                delta_ccz = 0.

            circuit = qutrit_toffoli_decomposition_circuit(node.op.name, rcr_type, qids,
                                                           self.inst_durations,
                                                           apply_dd=self.apply_dd,
                                                           pulse_alignment=self.pulse_alignment,
                                                           delta_cz=delta_cz, delta_ccz=delta_ccz)

            dag.substitute_node_with_dag(node, circuit_to_dag(circuit))

        return dag


def reverse2q_3q_decomposition_circuit(
    gate: str,
    physical_qubits: tuple[int, int, int],
    instruction_durations: InstructionDurations,
    apply_dd: bool = True,
    pulse_alignment: int = 1,
    delta_cz: float = 0.,
    include_last_local: bool = True
) -> QuantumCircuit:
    """Return the decomposition of the CZ part of the CCZ sequence."""
    circuit = QuantumCircuit(3)
    r2q = reverse2q_decomposition_circuit(gate, physical_qubits[1:], instruction_durations,
                                          apply_dd=apply_dd, pulse_alignment=pulse_alignment,
                                          delta_cz=delta_cz,
                                          include_last_local=include_last_local)
    circuit.compose(r2q, qubits=[1, 2], inplace=True)

    if apply_dd:
        def dur(gate, *qubits):
            return instruction_durations.get(gate, tuple(physical_qubits[iq] for iq in qubits))

        ddapp = DDCalculator(physical_qubits, instruction_durations, pulse_alignment)

        x12_to_x12 = dur('x12', 1) + 2 * dur('ecr', 2, 1) + dur('x', 2)
        xplus_dur = dur('x12', 1) + dur('x', 1)

        if gate == 'qutrit_qubit_cx':
            circuit.delay(dur('sx', 2), 0)

        ddapp.append_dd(circuit, 0, 2 * x12_to_x12 + xplus_dur, 5)

        if gate == 'qutrit_qubit_cx':
            circuit.delay(dur('sx', 2), 0)

    return circuit


def forwardcx_3q_decomposition_circuit(
    physical_qubits: tuple[int, int, int],
    instruction_durations: InstructionDurations,
    rcr_type: QutritQubitCXType,
    apply_dd: bool = True,
    pulse_alignment: int = 1
) -> QuantumCircuit:
    """Return the CX circuit using CRCR."""
    def dur(gate, *qubits):
        return instruction_durations.get(gate, tuple(physical_qubits[iq] for iq in qubits))

    circuit = QuantumCircuit(3)
    circuit.append(QutritQubitCXGate(), [1, 2])
    circuit.sx(1)
    circuit.append(P2Gate(np.pi / 4.), 1)

    if apply_dd:
        ddapp = DDCalculator(physical_qubits, instruction_durations, pulse_alignment)
        ddapp.append_dd(circuit, 0, dur(f'qutrit_qubit_cx_rcr{rcr_type}', 1, 2), 3)

    return circuit


def qutrit_toffoli_decomposition_circuit(
    gate: str,
    rcr_type: QutritQubitCXGate,
    physical_qubits: tuple[int, int, int],
    instruction_durations: InstructionDurations,
    apply_dd: bool = True,
    pulse_alignment: int = 1,
    refocusing_delay: int = 0,
    delta_cz: float = 0.,
    delta_ccz: float = 0.
) -> QuantumCircuit:
    """Return the decomposition of qubit-qutrit-qubit CCZ."""
    def dur(gate, *qubits):
        return instruction_durations.get(gate, tuple(physical_qubits[iq] for iq in qubits))

    xplus_dur = dur('x12', 1) + dur('x', 1)

    circuit = QuantumCircuit(3)

    if apply_dd:
        ddapp = DDCalculator(physical_qubits, instruction_durations, pulse_alignment)

    if (rcr_type == QutritQubitCXType.REVERSE) == (gate == 'qutrit_toffoli'):
        circuit.delay(dur('sx', 2), [0, 1])
        circuit.h(2)

    circuit.append(XplusGate(label='qutrit_toffoli_begin'), [1])
    if refocusing_delay:
        circuit.delay(refocusing_delay, 1)
    circuit.append(X12Gate(), [1])
    circuit.sx(1)
    circuit.append(P2Gate(-np.pi / 4.), [1])

    if apply_dd:
        ddapp.append_dd(circuit, 0, 2 * xplus_dur + refocusing_delay, distribution='left')
        ddapp.append_dd(circuit, 2, 2 * xplus_dur + refocusing_delay, distribution='right')
    else:
        circuit.delay(2 * xplus_dur + refocusing_delay, [0, 2])

    circuit.barrier()  # For refocusing analysis
    circuit.ecr(0, 1)

    if apply_dd:
        ddapp.append_dd(circuit, 2, dur('ecr', 0, 1) + dur('x12', 1), 2)
    else:
        circuit.delay(dur('ecr', 0, 1) + dur('x12', 1), 2)

    if rcr_type == QutritQubitCXType.REVERSE:
        cz = reverse2q_3q_decomposition_circuit('qutrit_qubit_cz', physical_qubits,
                                                instruction_durations, apply_dd=apply_dd,
                                                pulse_alignment=pulse_alignment,
                                                delta_cz=delta_cz, include_last_local=False)
        circuit.compose(cz, inplace=True)
        circuit.append(X12Gate(), [1])
        circuit.rz(np.pi, 1)
        circuit.sx(1)
        circuit.rz(-np.pi, 1)
        circuit.append(P2Gate(3. * np.pi / 4.), [1])
        if apply_dd:
            circuit.x(2)
            circuit.delay(dur('sx', 1), 2)
        else:
            circuit.delay(dur('x', 2) + dur('sx', 1), 2)
    else:
        cx = forwardcx_3q_decomposition_circuit(physical_qubits, instruction_durations, rcr_type,
                                                apply_dd=apply_dd, pulse_alignment=pulse_alignment)
        circuit.compose(cx, inplace=True)

    circuit.ecr(0, 1)

    circuit.delay(xplus_dur, 0)
    circuit.append(XplusGate('qutrit_toffoli_end'), [1])
    if delta_ccz:
        circuit.rz(-delta_ccz, 1)

    if apply_dd:
        ddapp.append_dd(circuit, 2, dur('ecr', 0, 1) + dur('x12', 1), 2)
        circuit.x(2)
    else:
        circuit.delay(dur('x', 2) + dur('ecr', 0, 1) + dur('x12', 1), 2)

    return circuit


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
    If 0 < d + s < a, we can achieve the above condition by adjusting some delays. For all three
    cases above, s = -3α where α is the duration of a sequence that is repeated twice. By adding a
    delay Δ to this sequence,
        d -> d' = d + 2Δ
        s -> s' = s - 3Δ
        ∴ d' + s' = d + s - Δ
    We are out of luck if d + s < 0.

    Since we introduce new gates and delays, node_start_time will be invalidated in this pass.
    """
    def __init__(
        self,
        instruction_durations: InstructionDurations,
        apply_dd: bool,
        pulse_alignment: int
    ):
        super().__init__()
        self.inst_durations = instruction_durations
        self.apply_dd = apply_dd
        self.pulse_alignment = pulse_alignment
        self.calibrations = None

    def _refocus_toffoli(
        self,
        dag,
        begin_node,
    ):
        node_start_time = self.property_set['node_start_time']

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

        rcr_type = self.calibrations.get_parameter_value('rcr_type', qids[1:])

        subdag = DAGCircuit()
        qreg = QuantumRegister(3)
        subdag.add_qreg(qreg)

        def dur(gate, *iqs):
            return self.inst_durations.get(gate, [qids[i] for i in iqs])

        def add_op(op, *iqs):
            subdag.apply_operation_back(op, [qreg[iq] for iq in iqs])

        def add_delay(delay, *iqs):
            add_op(Delay(delay), *iqs)

        if any((isinstance(node, DAGOpNode) and isinstance(node.op, HGate))
                for node in dag.ancestors(barrier)):
            add_delay(dur('sx', 2), 0)
            add_delay(dur('sx', 2), 1)
            add_op(HGate(), 2)

        if rcr_type == QutritQubitCXType.REVERSE:
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
            if (delay := added_time - dur('xplus', 1)):
                add_delay(delay, 1)
            add_op(X12Gate(), 1)
            add_op(SXGate(), 1)
            add_op(P2Gate(-np.pi / 4.), 1)
            if self.apply_dd:
                ddcalc = DDCalculator(qids, self.inst_durations, self.pulse_alignment)
                taus = ddcalc.calculate_delays(0, added_time + dur('x12', 1) + dur('sx', 1),
                                               distribution='left')
                for tau in taus[1:]:
                    add_op(XGate(), 0)
                    add_delay(tau, 0)
                taus = ddcalc.calculate_delays(2, added_time + dur('x12', 1) + dur('sx', 1),
                                               distribution='right')
                for tau in taus[:-1]:
                    add_delay(tau, 2)
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        toffoli_begins = [node for node in dag.topological_op_nodes()
                          if node.op.label == 'qutrit_toffoli_begin']

        for begin_node in toffoli_begins:
            self._refocus_toffoli(dag, begin_node)

        # Invalidate node_start_time
        self.property_set.pop('node_start_time')

        return dag
