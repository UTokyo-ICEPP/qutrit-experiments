from collections.abc import Sequence
from typing import Optional
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate
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
