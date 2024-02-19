"""Transpiler passses dealing with Rz angles."""
import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit.library import RZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass
import rustworkx as rx

from ..constants import LO_SIGN
from ..gates import RZ12Gate

twopi = 2. * np.pi


class InvertRZSign(TransformationPass):
    """Transpiler pass to invert all RZGate and RZ12Gate signs."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in list(dag.topological_op_nodes()):
            if isinstance(node.op, (RZGate, RZ12Gate)):
                # Fix the sign to make the gate correspond to its intended physical operation
                node.op.params[0] *= -LO_SIGN
        return dag


class ConsolidateRZAngle(TransformationPass):
    """Sum up the angles of neighboring Rz instructions."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set.get('node_start_time')

        def filter_fn(node):
            return (
                isinstance(node, DAGOpNode)
                and len(node.qargs) == 1
                and len(node.cargs) == 0
                and isinstance(node.op, RZGate)
                and not node.op.is_parameterized()
            )

        for run in rx.collect_runs(dag._multi_graph, filter_fn):
            angle = sum(node.op.params[0] for node in run) % twopi
            subdag = DAGCircuit()
            subdag.add_qreg((qreg := QuantumRegister(len(run[0].qargs))))
            subdag.apply_operation_back(RZGate(angle), [qreg[0]])
            subst_map = dag.substitute_node_with_dag(run[0], subdag)
            # Delete the other nodes in the run
            for node in run[1:]:
                dag.remove_op_node(node)
            # Update the node_start_time map if available
            if node_start_time:
                start_time = node_start_time.pop(run[0])
                op_nodes = tuple(subdag.topological_op_nodes())
                node_start_time[subst_map[op_nodes[0]._node_id]] = start_time

        return dag
