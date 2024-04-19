from collections.abc import Sequence
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Delay, Gate, Qubit
from qiskit.circuit.library import RZGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler.basepasses import AnalysisPass


def insert_rz(
    dag: DAGCircuit,
    node: DAGOpNode,
    pre_angles: Optional[Sequence[float]] = None,
    post_angles: Optional[Sequence[float]] = None,
    node_start_time: Optional[dict[DAGOpNode, int]] = None,
    op_duration: int = 0
) -> DAGOpNode:
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
    # Update the node_start_time map
    start_time = node_start_time.pop(node)
    op_nodes = iter(subdag.topological_op_nodes())
    for angle in pre_angles:
        if angle:
            node_start_time[subst_map[next(op_nodes)._node_id]] = start_time
    new_node = subst_map[next(op_nodes)._node_id]
    node_start_time[new_node] = start_time
    for angle in post_angles:
        if angle:
            node_start_time[subst_map[next(op_nodes)._node_id]] = start_time + op_duration

    return new_node


def insert_dd(
    subdag: DAGCircuit,
    qubit: Qubit,
    duration: int,
    x_duration: int,
    pulse_alignment: int,
    start_time: Optional[int] = None,
    placement: str = 'left'
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

    if (duration / 2) % pulse_alignment == 0:
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
    else:
        add_op(Gate(f'dd_{placement}', 1, [duration]), [qubit], 0)

    return start_times
