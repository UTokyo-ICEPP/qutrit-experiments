import logging
import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Delay
from qiskit.circuit.library import XGate
from qiskit.transpiler import (AnalysisPass, InstructionDurations, PassManager, Target,
                               TransformationPass, TranspilerError)
from qiskit_experiments.calibration_management import Calibrations
import rustworkx as rx

from ..gates import QutritQubitCXGate, X12Gate, XplusGate, XminusGate

logger = logging.getLogger(__name__)


class QutritToffoliRefocusing(TransformationPass):
    """Calculate the phase errors due to f12 detuning and convert the last X+ to X-X- with an
    inserted delay."""
    def __init__(self):
        super().__init__()
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']

        for node in list(dag.topological_op_nodes()):
            if (start_time := node_start_time.get(node)) is None:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            if isinstance(node.op, XminusGate) and node.op.label == 'qutrit_toffoli_begin':
                toffoli_begin = start_time
                subdag = DAGCircuit()
                qreg = QuantumRegister(1)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(XplusGate(), [qreg[0]])
                subdag.apply_operation_back((refocusing_delay := Delay(0)), [qreg[0]])
                subdag.apply_operation_back(XplusGate(), [qreg[0]])
                subst_map = dag.substitute_node_with_dag(node, subdag)

                # Save the first two nodes and assign the current start time to the last
                node_start_time.pop(node)
                op_nodes = iter(subdag.topological_op_nodes())
                first_xplus_node = subst_map[next(op_nodes)._node_id]
                delay_node = subst_map[next(op_nodes)._node_id]
                node_start_time[subst_map[next(op_nodes)._node_id]] = start_time
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

