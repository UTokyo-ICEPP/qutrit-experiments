"""Common utility functions for qutrit transpilation."""
from collections.abc import Sequence
from typing import Optional
from qiskit import QuantumRegister
from qiskit.circuit import Delay, Gate, Qubit
from qiskit.circuit.library import RZGate, XGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode


def insert_rz(
    dag: DAGCircuit,
    node: DAGOpNode,
    pre_angles: Optional[Sequence[float]] = None,
    post_angles: Optional[Sequence[float]] = None,
    node_start_time: Optional[dict[DAGOpNode, int]] = None,
    op_duration: int = 0
) -> DAGOpNode:
    """Insert an Rz gate to the DAG before and after the given node."""
    subdag = DAGCircuit()
    subdag.add_qreg((qreg := QuantumRegister(len(node.qargs))))
    if pre_angles is None:
        pre_angles = [0.] * qreg.size
    if post_angles is None:
        post_angles = [0.] * qreg.size
    for qubit, angle in zip(qreg, pre_angles):
        if angle:
            subdag.apply_operation_back(RZGate(angle), [qubit])
    subdag.apply_operation_back(node.op, tuple(qreg))
    for qubit, angle in zip(qreg, post_angles):
        if angle:
            subdag.apply_operation_back(RZGate(angle), [qubit])
    subst_map = dag.substitute_node_with_dag(node, subdag)
    op_nodes = list(subdag.topological_op_nodes())
    new_node_idx = sum(1 for angle in pre_angles if angle)
    new_node = subst_map[op_nodes[new_node_idx]._node_id]

    # Update the node_start_time map
    if node_start_time:
        start_time = node_start_time.pop(node)
        node_start_time[new_node] = start_time
        for node in op_nodes[:new_node_idx]:
            node_start_time[subst_map[node._node_id]] = start_time
        for node in op_nodes[new_node_idx + 1:]:
            node_start_time[subst_map[node._node_id]] = start_time + op_duration

    return new_node


def insert_dd(
    subdag: DAGCircuit,
    qubit: Qubit,
    duration: int,
    x_duration: int,
    pulse_alignment: int,
    start_time: Optional[int] = None,
    placement: str = 'symmetric'
) -> bool:
    """Insert either two X gates or a generic gate named dd_left/right.

    Returns True if the generic gate is inserted.
    """
    start_times = None if start_time is None else []

    if duration < 2 * x_duration:
        return start_times

    def add_op(gate, qargs, offset):
        node = subdag.apply_operation_back(gate, qargs)
        if start_time is not None:
            start_times.append((node, start_time + offset))

    if placement in ['left', 'right'] and (duration / 2) % pulse_alignment == 0:
        interval = duration // 2 - x_duration
        offset = 0
        if placement == 'left':
            add_op(XGate(), [qubit], 0)
            offset += x_duration
        if interval != 0:
            add_op(Delay(interval), [qubit], offset)
            offset += interval
        add_op(XGate(), [qubit], offset)
        offset += x_duration
        if interval != 0:
            add_op(Delay(interval), [qubit], offset)
            offset += interval
        if placement == 'right':
            add_op(XGate(), [qubit], offset)
    elif placement == 'symmetric' and (duration / 4 - x_duration / 2) % pulse_alignment == 0:
        interval = int(duration / 4 - x_duration / 2)
        offset = 0
        add_op(Delay(interval), [qubit], offset)
        offset += interval
        add_op(XGate(), [qubit], offset)
        offset += x_duration
        add_op(Delay(interval * 2), [qubit], offset)
        offset += interval * 2
        add_op(XGate(), [qubit], offset)
        offset += x_duration
        add_op(Delay(interval), [qubit], offset)
    else:
        add_op(Gate(f'dd_{placement}', 1, [duration]), [qubit], 0)

    return start_times
