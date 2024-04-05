from dataclasses import dataclass
import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Barrier, Delay, Gate, Qubit
from qiskit.circuit.library import CXGate, XGate
from qiskit.transpiler import Target, TransformationPass, TranspilerError

from ..calibrations import get_qutrit_qubit_composite_gate
from ..gates import QutritQubitCXGate, QutritToffoliGate, XplusGate, XminusGate


logger = logging.getLogger(__name__)


class QutritToffoliDecomposition(TransformationPass):
    def __init__(self, target: Target):
        super().__init__()
        self.target = target
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritToffoliGate):
                qids = tuple(dag.find_bit(q).index for q in node.qargs)
                rcr_type = self.calibrations.get_parameter_value('rcr_type', qids[1:])
                cx_gate = QutritQubitCXGate.of_type(rcr_type)

                subdag = DAGCircuit()
                qreg = QuantumRegister(3)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(Barrier(3), qreg)
                subdag.apply_operation_back(XminusGate(label='qutrit_toffoli_begin'), [qreg[1]])
                subdag.apply_operation_back(Barrier(2), qreg[:2])
                subdag.apply_operation_back(CXGate(), qreg[:2])
                subdag.apply_operation_back(Barrier(3), qreg)
                subdag.apply_operation_back(cx_gate(), qreg[1:])
                subdag.apply_operation_back(Barrier(3), qreg)
                subdag.apply_operation_back(CXGate(), qreg[:2])
                subdag.apply_operation_back(Barrier(2), qreg[:2])
                subdag.apply_operation_back(XplusGate(label='qutrit_toffoli_end'), [qreg[1]])
                subdag.apply_operation_back(Barrier(3), qreg)
                dag.substitute_node_with_dag(node, subdag)

                if dag.calibrations.get(cx_gate.gate_name, {}).get(qids[1:], ()) is None:
                    sched = get_qutrit_qubit_composite_gate(cx_gate.gate_name, qids[1:],
                                                            self.calibrations, target=self.target)
                    dag.add_calibration(cx_gate.gate_name, qids[1:], sched)

        return dag


class QutritToffoliRefocusing(TransformationPass):
    """Calculate the phase errors due to f12 detuning and convert the last X+ to X-X- with an
    inserted delay.

    At the current form this pass is specific to Toffoli testing and cannot be used in real contexts
    where the gate is used within some logical circuit."""
    def __init__(self):
        super().__init__()
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']

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
                    delay_node = next(dag.successors(node))
                    if not isinstance(delay_node.op, Delay):
                        raise TranspilerError('X+(qutrit_toffoli_begin) is not followed by a delay')
                    refocusing_delay = delay_node.op
            elif isinstance(node.op, XplusGate) and node.op.label == 'qutrit_toffoli_end':
                toffoli_end = start_time
            elif isinstance(node.op, QutritQubitCXGate):
                cx_qubits = tuple(dag.find_bit(q).index for q in node.qargs)

        x_duration = self.calibrations.get_schedule('x', cx_qubits[0]).duration
        x12_duration = self.calibrations.get_schedule('x12', cx_qubits[0]).duration
        cr_duration = self.calibrations.get_schedule('cr', cx_qubits).duration

        # Phase error due to f12 detuning (detuning factored out)
        u_qutrit_phase = np.array([0., 0., -3. * (x_duration + x12_duration + 2. * cr_duration)])
        added_time = u_qutrit_phase[2] + (toffoli_end - toffoli_begin)
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
    qutrit_qubit_cx: DAGOpNode


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
            qutrit_qubit_cx = node
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
                    barriers=tuple(barriers),
                    qutrit_qubit_cx=qutrit_qubit_cx
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
            sched = get_qutrit_qubit_composite_gate(f'{info.qutrit_qubit_cx.name}_dd', qids,
                                                    self.calibrations, target=self.target)
            dag.add_calibration(info.qutrit_qubit_cx.name, qids[1:], sched)

        return dag
