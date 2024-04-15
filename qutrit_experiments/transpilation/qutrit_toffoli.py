"""Transpiler passes to convert the Qutrit Toffoli gate into a fully scheduled circuit.

All passes in this module are meant to be run on a circuit that contains one Toffoli gate and
nothing else.
"""
from dataclasses import dataclass
import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Barrier, Delay, Gate, Qubit
from qiskit.circuit.library import CXGate, ECRGate, HGate, RZGate, XGate
from qiskit.transpiler import InstructionDurations, Target, TransformationPass, TranspilerError

from ..calibrations import get_qutrit_qubit_composite_gate
from ..gates import (QutritQubitCXGate, QutritQubitCXTypeReverseGate, QutritToffoliGate, RZ12Gate,
                     XplusGate, XminusGate)

logger = logging.getLogger(__name__)


class QutritToffoliDecomposition(TransformationPass):
    """Decompose QutritToffoliGate to the basic sequence and append the CX calibration if
    applicable."""
    def __init__(self, target: Target, instruction_durations: InstructionDurations):
        super().__init__()
        self.target = target
        self.inst_durations = instruction_durations
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, QutritToffoliGate):
                continue

            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids[1:])
            reverse_cx = (rcr_type == QutritQubitCXGate.TYPE_REVERSE)
            cx_gate = QutritQubitCXGate.of_type(rcr_type)

            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)
            subdag.apply_operation_back(XminusGate(label='qutrit_toffoli_begin'), [qreg[1]])
            subdag.apply_operation_back(Barrier(2), qreg[:2])
            subdag.apply_operation_back(CXGate(), qreg[:2])
            subdag.apply_operation_back(Barrier(3), qreg)
            if reverse_cx:
                self.insert_reverse_cx(subdag, qids)
            else:
                subdag.apply_operation_back(cx_gate(), qreg[1:])
            subdag.apply_operation_back(Barrier(3), qreg)
            subdag.apply_operation_back(CXGate(), qreg[:2])
            subdag.apply_operation_back(Barrier(2), qreg[:2])
            subdag.apply_operation_back(XplusGate(label='qutrit_toffoli_end'), [qreg[1]])
            dag.substitute_node_with_dag(node, subdag)

            if (not reverse_cx
                and dag.calibrations.get(cx_gate.gate_name, {}).get(qids[1:], ()) is None):
                sched = get_qutrit_qubit_composite_gate(cx_gate.gate_name, qids[1:],
                                                        self.calibrations, target=self.target)
                dag.add_calibration(cx_gate.gate_name, qids[1:], sched)

        return dag
    
    def insert_reverse_cx(self, subdag: DAGCircuit, qids: tuple[int, int, int]):
        def dur(gate, *iqs):
            return self.inst_durations.get(gate, [qids[i] for i in iqs])

        qreg = next(iter(subdag.qregs.values()))
        t = 0
        subdag.apply_operation_back(XplusGate(), [qreg[1]])
        subdag.apply_operation_back(HGate(), [qreg[2]])
        subdag.apply_operation_back(XGate(), [qreg[2]])
        t += max(dur('xplus', 1), dur('sx', 2) + dur('x', 2))
        subdag.apply_operation_back(ECRGate(), [qreg[2], qreg[1]])
        t += dur('ecr', 2, 1)
        subdag.apply_operation_back(XGate(), [qreg[1]])
        subdag.apply_operation_back(RZGate(np.pi / 3.), [qreg[1]]) # Cancel the geometric phase correction
        subdag.apply_operation_back(RZ12Gate(2. * np.pi / 3.), [qreg[1]]) # Cancel the geometric phase correction
        subdag.apply_operation_back(XGate(), [qreg[2]])
        t += max(dur('x', 1), dur('x', 2))
        subdag.apply_operation_back(ECRGate(), [qreg[2], qreg[1]])
        t += dur('ecr', 2, 1)
        subdag.apply_operation_back(XplusGate(), [qreg[1]])
        subdag.apply_operation_back(RZGate(np.pi), [qreg[2]])
        subdag.apply_operation_back(HGate(), [qreg[2]])
        subdag.apply_operation_back(Delay(t - dur('xplus', 1)), [qreg[1]])
        subdag.apply_operation_back(XplusGate(label='revese_cx_end'), [qreg[1]])


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
    def __init__(self):
        super().__init__()
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']

        cx_qubits = None
        for node in dag.topological_op_nodes():
            if (start_time := node_start_time.get(node)) is None:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            if node.op.label == 'qutrit_toffoli_begin':
                if isinstance(node.op, XminusGate):
                    # Convert to X+-delay-X+
                    toffoli_begin = start_time
                    qutrit = dag.find_bit(node.qargs[0]).index
                    subdag = DAGCircuit()
                    qreg = QuantumRegister(1)
                    subdag.add_qreg(qreg)
                    subdag.apply_operation_back(XplusGate(label='qutrit_toffoli_begin'), [qreg[0]])
                    subdag.apply_operation_back((refocusing_delay := Delay(0)), [qreg[0]])
                    subdag.apply_operation_back(XplusGate(), [qreg[0]])
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
                cx_delay_node = next(dag.predecessors(node))
            elif isinstance(node.op, QutritQubitCXGate):
                cx_qubits = tuple(dag.find_bit(q).index for q in node.qargs)

        x_duration = self.calibrations.get_schedule('x', qutrit).duration
        x12_duration = self.calibrations.get_schedule('x12', qutrit).duration
        # Phase error due to f12 detuning (detuning factored out)
        if cx_qubits:
            cr_duration = self.calibrations.get_schedule('cr', cx_qubits).duration
            alpha = x_duration + x12_duration + 2. * cr_duration
        else:
            alpha = x_duration + x12_duration + cx_delay_node.duration

        added_time = -3 * alpha + (toffoli_end - toffoli_begin)
        # Subtract the X+ duration
        refocusing_delay.duration = added_time - (x_duration + x12_duration)

        # Update all node start times
        for node in dag.topological_op_nodes():
            if node in (first_xplus_node, delay_node):
                continue
            node_start_time[node] += added_time

        node_start_time[first_xplus_node] = toffoli_begin
        node_start_time[delay_node] = toffoli_begin + x_duration + x12_duration

        return dag


@dataclass
class QutritToffoliInfo:
    qubits: tuple[Qubit, Qubit, Qubit]
    xplus_times: tuple[int, int]
    barriers: tuple[DAGOpNode, ...]


class QutritToffoliDynamicalDecoupling(TransformationPass):
    """Insert DD sequences to idle qubits. More aggressive than PadDynamicalDecoupling."""
    def __init__(self, target: Target):
        super().__init__()
        self.target = target
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set["node_start_time"]

        # Analyze the gate structure (qargs) first
        # Store for all toffoli info
        toffoli_info = []
        for xplus_node in dag.named_nodes('xplus'):
            if xplus_node.op.label != 'qutrit_toffoli_begin':
                continue
            c2_qubit = xplus_node.qargs[0]
            logger.debug('xplus0 on %s found at t=%d', c2_qubit, node_start_time[xplus_node])
            xplus_times = [node_start_time[xplus_node]]
            barriers = [next(dag.predecessors(xplus_node))]
            # Follow the graph for the next X+
            node = xplus_node
            while True:
                if isinstance((node := next(dag.successors(node))).op, XplusGate):
                    break
            logger.debug('xplus1 found at t=%d', node_start_time[node])
            xplus_times.append(node_start_time[node])
            barriers.append(next(dag.successors(node)))
            # Keep following the graph until we hit a two-qubit gate
            while True:
                if len((node := next(dag.successors(node))).qargs) == 2:
                    break
            logger.debug('CX/ECR(0, 1) on %s found at t=%d', node.qargs, node_start_time[node])
            c1_qubit = node.qargs[0]
            # Keep following the graph until we hit a barrier
            while True:
                if isinstance((node := next(dag.successors(node))).op, Barrier):
                    break
            barriers.append(node)
            node = next(s for s in dag.successors(node) if c1_qubit not in s.qargs)
            logger.debug('CX(1, 2) on %s found at t=%d', node.qargs, node_start_time[node])
            t_qubit = node.qargs[1]
            node = next(dag.successors(node))
            barriers.append(node)
            # Follow until next barrier
            while True:
                if isinstance((node := next(dag.successors(node))).op, Barrier):
                    break
            barriers.append(node)
            # Find the final barrier
            while True:
                if isinstance((node := next(dag.successors(node))).op, Barrier):
                    break
            barriers.append(node)

            toffoli_info.append(
                QutritToffoliInfo(
                    qubits=(c1_qubit, c2_qubit, t_qubit),
                    xplus_times=tuple(xplus_times),
                    barriers=tuple(barriers)
                )
            )

        for info in toffoli_info:
            qids = tuple(dag.find_bit(q).index for q in info.qubits)
            logger.debug('%s', qids)
            xplus_duration = sum(self.calibrations.get_schedule(gate, qids[1]).duration
                                 for gate in ['x', 'x12'])
            x_durations = [self.target['x'][(qid,)].calibration.duration for qid in qids]

            def add_dd(name, qubit, start_time, duration):
                if duration < 2 * x_durations[qubit]:
                    return
                if (duration / 2) % self.target.pulse_alignment == 0:
                    interval = duration // 2 - x_durations[qubit]
                    node = subdag.apply_operation_back(XGate(), [qreg[qubit]])
                    start_times.append((node, start_time))
                    if interval != 0:
                        node = subdag.apply_operation_back(Delay(interval), [qreg[qubit]])
                        start_times.append((node, start_time + x_durations[qubit]))
                    node = subdag.apply_operation_back(XGate(), [qreg[qubit]])
                    start_times.append((node, start_time + x_durations[qubit] + interval))
                    if interval != 0:
                        node = subdag.apply_operation_back(Delay(interval), [qreg[qubit]])
                        start_times.append((node, start_time + 2 * x_durations[qubit] + interval))
                    return

                sched = self.calibrations.get_schedule('dd', qids[qubit],
                                                       assign_params={'duration': duration})
                dag.add_calibration(name, [qids[qubit]], sched)
                node = subdag.apply_operation_back(Gate(name, 1, []), [qreg[qubit]])
                start_times.append((node, start_time))

            # C1&T DD (begin side)
            start_time = node_start_time.pop(info.barriers[0])
            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)
            start_times = []

            node = subdag.apply_operation_back(Barrier(3), qreg)
            start_times.append((node, start_time))
            for sequence, time, duration in [
                ('xplus0', start_time, xplus_duration),
                ('delay', start_time + xplus_duration,
                 info.xplus_times[1] - (info.xplus_times[0] + xplus_duration)),
                ('xplus1', node_start_time[info.barriers[1]] - xplus_duration, xplus_duration)
            ]:
                add_dd(f'c1_dd_{sequence}', 0, time, duration)
                add_dd(f't_dd_{sequence}', 2, time, duration)
            add_dd('t_dd_cx0', 2, node_start_time[info.barriers[1]],
                   node_start_time[info.barriers[2]] - node_start_time[info.barriers[1]])

            subst_map = dag.substitute_node_with_dag(info.barriers[0], subdag)

            for node, time in start_times:
                node_start_time[subst_map[node._node_id]] = time

            # C1&T DD (end side)
            end_time = node_start_time.pop(info.barriers[-1])
            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)
            start_times = []

            add_dd('t_dd_cx1', 2, node_start_time[info.barriers[-3]],
                   node_start_time[info.barriers[-2]] - node_start_time[info.barriers[-3]])
            add_dd('c1_dd_xplus2', 0, node_start_time[info.barriers[-2]], xplus_duration)
            add_dd('t_dd_xplus2', 2, node_start_time[info.barriers[-2]], xplus_duration)
            node = subdag.apply_operation_back(Barrier(3), qreg)
            start_times.append((node, end_time))

            subst_map = dag.substitute_node_with_dag(info.barriers[-1], subdag)

            for node, time in start_times:
                node_start_time[subst_map[node._node_id]] = time

            # C1 DD (during qutrit-qubit CX)
            start_time = node_start_time.pop(info.barriers[2])
            subdag = DAGCircuit()
            qreg = QuantumRegister(3)
            subdag.add_qreg(qreg)
            start_times = []

            node = subdag.apply_operation_back(Barrier(3), qreg)
            start_times.append((node, start_time))

            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids[1:])
            cx_gate_name = QutritQubitCXGate.of_type(rcr_type).gate_name
            name = f'{cx_gate_name}_dd'
            sched = self.calibrations.get_schedule(name, qids[0])
            dag.add_calibration(name, [qids[0]], sched)
            node = subdag.apply_operation_back(Gate(name, 1, []), [qreg[0]])
            start_times.append((node, start_time))

            subst_map = dag.substitute_node_with_dag(info.barriers[2], subdag)

            for node, time in start_times:
                node_start_time[subst_map[node._node_id]] = time

        return dag
