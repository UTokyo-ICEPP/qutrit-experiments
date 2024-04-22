"""Transpiler passes to convert the Qutrit Toffoli gate into a fully scheduled circuit.

All passes in this module are meant to be run on a circuit that contains one Toffoli gate and
nothing else.
"""
import logging
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.transpiler import (AnalysisPass, InstructionDurations, Target, TransformationPass,
                               TranspilerError)

from ..gates import (QutritQubitCXType, QutritQubitCXGate, QutritMCGate, XplusGate,
                     XminusGate)
from .util import insert_dd

logger = logging.getLogger(__name__)


class ContainsQutritMCGate(AnalysisPass):
    """Search the DAG circuit for qutrit Toffoli gates."""
    def run(self, dag: DAGCircuit):
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritMCGate):
                self.property_set['has_qutrit_mcgate'] = True
                return


class QutritToffoliRefocusing(TransformationPass):
    """Calculate the phase errors due to f12 detuning and convert the last X+ to X-X- with an
    inserted delay.

    With f12 detuning, X12->X12 exp(-iδtζ). Effect on CX(c2, t) is a phase shift on the |2> level.
    Type CRCR_X
        CR+ X CR+ X12(t2) CR+ X CR+ X12(t1) CR- X CR- X12(t0)
      = CR+ X CR+ X12 CR+ X CR+ X12 CR- X CR- X12 exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CX exp[iδα(z+2ζ)] ~ CX P2(-3α)
    where α is the time between the two X12s
    Type CRCR_X12
        X CR- X12(t2) CR- X CR+ X12(t1) CR+ X CR+ X12(t0) CR+
      = X CR- X12 CR- X CR+ X12 CR+ X CR+ X12 CR+ exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CX exp[iδα(z+2ζ)] ~ CX P2(-3α)
    Type REVERSE
        X+(t2) Delay X+(t1) ECR X ECR X+(t0)
      = X+ Delay X+ ECR X ECR X+ exp[-iδ(t2(-z-ζ) + t1z + t0ζ)]
      = CX exp[iδα(z+2ζ)] ~ CX P2(-3α)
    where α is the time between the first two X12s

    """
    def __init__(self, instruction_durations: InstructionDurations):
        super().__init__()
        self.inst_durations = instruction_durations

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']
        time_unit = self.property_set['time_unit']

        cx_qubits = None
        for node in dag.topological_op_nodes():
            if (start_time := node_start_time.get(node)) is None:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            if node.op.label == 'qutrit_toffoli_begin':
                toffoli_begin = start_time
                if isinstance(node.op, XminusGate):
                    # Convert to X+-delay-X+
                    qutrit = dag.find_bit(node.qargs[0]).index
                    subdag = DAGCircuit()

                    def add_gate(op):
                        node = subdag.apply_operation_back(op, [qreg[0]])
                        node.op = node.op.to_mutable()
                        node.op.duration = self.inst_durations.get(node.op, [qutrit], unit=time_unit)
                        node.op.unit = time_unit

                    qreg = QuantumRegister(1)
                    subdag.add_qreg(qreg)
                    add_gate(XplusGate(label='qutrit_toffoli_begin'))
                    add_gate((refocusing_delay := Delay(0)))
                    add_gate(XplusGate())
                    subst_map = dag.substitute_node_with_dag(node, subdag)

                    # Save the first two nodes and assign the current start time to the last
                    node_start_time.pop(node)
                    op_nodes = iter(subdag.topological_op_nodes())
                    first_xplus_node = subst_map[next(op_nodes)._node_id]
                    delay_node = subst_map[next(op_nodes)._node_id]
                    node_start_time[subst_map[next(op_nodes)._node_id]] = start_time
                elif isinstance(node.op, XplusGate):
                    # Assume to already be in sequence X+-delay-X+
                    first_xplus_node = node
                    delay_node = next(dag.successors(node))
                    if not isinstance(delay_node.op, Delay):
                        raise TranspilerError('X+(qutrit_toffoli_begin) is not followed by a delay')
                    refocusing_delay = delay_node.op
            elif isinstance(node.op, XplusGate) and node.op.label == 'qutrit_toffoli_end':
                toffoli_end = start_time
            elif isinstance(node.op, XplusGate) and node.op.label == 'reverse_cx_end':
                cx_delay = next(dag.predecessors(node)).op.duration
            elif isinstance(node.op, QutritQubitCXGate):
                cx_qubits = tuple(dag.find_bit(q).index for q in node.qargs)

        x_duration = self.inst_durations.get('x', [qutrit])
        x12_duration = self.inst_durations.get('x12', [qutrit])
        # Phase error due to f12 detuning (detuning factored out)
        if cx_qubits:
            cr_duration = self.inst_durations.get('cr', cx_qubits)
            alpha = x_duration + x12_duration + 2. * cr_duration
        else:
            alpha = x_duration + x12_duration + cx_delay

        added_time = -3 * alpha + (toffoli_end - toffoli_begin)
        # Subtract the X+ duration
        refocusing_delay.duration = added_time - (x_duration + x12_duration)

        # Update all node start times
        not_shifted = dag.ancestors(delay_node) | {delay_node}
        for node in dag.topological_op_nodes():
            if node in not_shifted:
                continue
            node_start_time[node] += added_time

        node_start_time[first_xplus_node] = toffoli_begin
        node_start_time[delay_node] = toffoli_begin + x_duration + x12_duration

        return dag


class QutritToffoliDynamicalDecoupling(TransformationPass):
    """Insert DD sequences to idle qubits. More aggressive than PadDynamicalDecoupling."""
    def __init__(
        self,
        target: Target,
        instruction_durations: InstructionDurations,
        rcr_types: dict[tuple[int, int], int]
    ):
        super().__init__()
        self.target = target
        self.inst_durations = instruction_durations
        self.rcr_types = dict(rcr_types)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set["node_start_time"]

        # Analyze the gate structure (qargs) first
        first_barrier = next(node for node in dag.topological_op_nodes()
                             if isinstance(node.op, Barrier))
        circuit_qargs = first_barrier.qargs
        barriers = [first_barrier]
        # Get the first X+
        node = next(s for s in dag.successors(first_barrier) if isinstance(s.op, XplusGate))
        c2_qubit = node.qargs[0]
        logger.debug('xplus0 on %s found at t=%d', c2_qubit, node_start_time[node])
        xplus_times = [node_start_time[node]]
        # Follow the graph for the next X+
        while True:
            if isinstance((node := next(dag.successors(node))).op, XplusGate):
                break
        logger.debug('xplus1 found at t=%d', node_start_time[node])
        xplus_times.append(node_start_time[node])
        barriers.append(next(dag.successors(node)))
        # Follow the graph until we hit a two-qubit gate between c1 and c2
        while True:
            if len((node := next(dag.successors(node))).qargs) == 2:
                break
        logger.debug('CX/ECR(0, 1) on %s found at t=%d', node.qargs, node_start_time[node])
        c1_qubit = node.qargs[0]
        t_qubit = next(q for q in circuit_qargs if q not in [c2_qubit, c1_qubit])
        # Follow the graph until we hit a barrier before c2-t CX
        while True:
            if isinstance((node := next(dag.successors(node))).op, Barrier):
                break
        barriers.append(node)
        # Follow the graph (skip the c2-t CX) until the next Barrier(3)
        # (need qargs condition because reverse cx uses a Barrier(2) internally)
        while True:
            if isinstance((node := next(dag.successors(node))).op, Barrier) and len(node.qargs) == 3:
                break
        barriers.append(node)
        # Follow until the next barrier(2)
        node = next(s for s in dag.successors(node) if t_qubit not in s.qargs)
        while True:
            if isinstance((node := next(dag.successors(node))).op, Barrier):
                break
        barriers.append(node)
        # Find the final barrier
        while True:
            if isinstance((node := next(dag.successors(node))).op, Barrier):
                break
        barriers.append(node)

        # Insert DD sequences
        qids = tuple(dag.find_bit(q).index for q in [c1_qubit, c2_qubit, t_qubit])
        logger.debug('qids %s', qids)
        xplus_duration = self.inst_durations.get('xplus', [qids[1]])
        x_durations = [self.inst_durations.get('x', [qid]) for qid in qids]
        rcr_type = self.rcr_types[qids[1:]]

        def make_dd_subdag():
            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)
            return subdag, qreg, []

        def add_dd(qubit, start_time, duration, placement='left'):
            times = insert_dd(subdag, qreg[qubit], duration, x_durations[qubit],
                              self.target.pulse_alignment, start_time=start_time,
                              placement=placement)
            start_times.extend(times)

        def insert_dd_to_dag(node_to_replace):
            subst_map = dag.substitute_node_with_dag(node_to_replace, subdag)
            for node, time in start_times:
                node_start_time[subst_map[node._node_id]] = time

        # C1&T DD (begin side)
        subdag, qreg, start_times = make_dd_subdag()

        node = subdag.apply_operation_back(Barrier(3), qreg)
        start_time = node_start_time.pop(barriers[0])
        start_times.append((node, start_time))
        for time, duration in [
            (start_time, xplus_duration),
            (start_time + xplus_duration, xplus_times[1] - (xplus_times[0] + xplus_duration)),
            (node_start_time[barriers[1]] - xplus_duration, xplus_duration)
        ]:
            add_dd(0, time, duration)
            add_dd(2, time, duration)
        add_dd(2, node_start_time[barriers[1]],
               node_start_time[barriers[2]] - node_start_time[barriers[1]], placement='right')

        insert_dd_to_dag(barriers[0])

        # C1&T DD (end side)
        subdag, qreg, start_times = make_dd_subdag()

        start_time = node_start_time[barriers[-3]]
        add_dd(2, start_time, node_start_time[barriers[-2]] - start_time)
        add_dd(0, node_start_time[barriers[-2]], xplus_duration)
        add_dd(2, node_start_time[barriers[-2]], xplus_duration)
        node = subdag.apply_operation_back(Barrier(3), qreg)
        start_times.append((node, node_start_time.pop(barriers[-1])))

        insert_dd_to_dag(barriers[-1])

        # C1 DD (during qutrit-qubit CX)
        subdag, qreg, start_times = make_dd_subdag()

        node = subdag.apply_operation_back(Barrier(3), qreg)
        start_time = node_start_time.pop(barriers[2])
        start_times.append((node, start_time))

        if rcr_type == QutritQubitCXType.REVERSE:
            ecr_duration = self.inst_durations.get('ecr', (qids[2], qids[1]))

            time = start_time
            add_dd(0, time, xplus_duration)
            time += xplus_duration
            add_dd(0, time, ecr_duration + x_durations[2], 'right')
            time += ecr_duration + x_durations[2]
            add_dd(0, time, ecr_duration + x_durations[2], 'right')
            time += ecr_duration + x_durations[2]
            # From the X of second X+ to just before the X12 of the last X+
            end_time = node_start_time[barriers[3]] - xplus_duration
            add_dd(0, time, end_time - time)
            time = end_time
            add_dd(0, time, xplus_duration)
        else:
            t_dd_duration = x_durations[2] * 2
            cr_duration = self.inst_durations.get('cr', qids[1:])
            sx_duration = self.inst_durations.get('sx', [qids[2]])
            time = start_time
            if rcr_type == QutritQubitCXType.X:
                for _ in range(3):
                    add_dd(0, time, t_dd_duration)
                    time += t_dd_duration
                    add_dd(0, time, cr_duration)
                    time += cr_duration
                    add_dd(0, time, t_dd_duration)
                    time += t_dd_duration
                    add_dd(0, time, cr_duration)
                    time += cr_duration
            else:
                for _ in range(3):
                    add_dd(0, time, cr_duration)
                    time += cr_duration
                    add_dd(0, time, t_dd_duration)
                    time += t_dd_duration
                    add_dd(0, time, cr_duration)
                    time += cr_duration
                    add_dd(0, time, t_dd_duration)
                    time += t_dd_duration
            add_dd(0, time, 2 * sx_duration) # cx_offset_rx duration

        insert_dd_to_dag(barriers[2])

        return dag
