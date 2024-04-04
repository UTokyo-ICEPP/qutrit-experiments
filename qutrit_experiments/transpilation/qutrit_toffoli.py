import logging
import numpy as np
from qiskit import QuantumRegister, pulse
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.circuit import Delay, Gate, Parameter
from qiskit.circuit.library import CXGate, XGate
from qiskit.transpiler import (AnalysisPass, InstructionDurations, PassManager, Target,
                               TransformationPass, TranspilerError)
from qiskit_experiments.calibration_management import Calibrations
import rustworkx as rx

from ..gates import QutritQubitCXGate, X12Gate, XplusGate, XminusGate
from ..calibrations.qutrit_qubit_cx import get_qutrit_qubit_composite_gate
from ..pulse_library import DoubleDrag

logger = logging.getLogger(__name__)


class QutritToffoliRefocusing(TransformationPass):
    """Calculate the phase errors due to f12 detuning and convert the last X+ to X-X- with an
    inserted delay."""
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

            if isinstance(node.op, XminusGate) and node.op.label == 'qutrit_toffoli_begin':
                toffoli_begin = start_time
                subdag = DAGCircuit()
                qreg = QuantumRegister(1)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(XplusGate(label='qutrit_toffoli_begin'), [qreg[0]])
                subdag.apply_operation_back((refocusing_delay := Delay(0)), [qreg[0]])
                subdag.apply_operation_back(XplusGate(label='qutrit_toffoli_refocusing'), [qreg[0]])
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


class QutritToffoliDynamicalDecoupling(TransformationPass):
    """Insert DD sequences to idle qubits. More aggressive than PadDynamicalDecoupling.

    At the current form this pass is specific to Toffoli testing and cannot be used in real contexts
    where the gate is used within some logical circuit.
    """
    def __init__(self, qubits: tuple[int, int, int], instruction_durations: InstructionDurations,
                 target: Target):
        super().__init__()
        self.qubits = qubits
        self.instruction_durations = instruction_durations
        self.target = target
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set["node_start_time"]

        first_c1_node = last_c1_node = None
        for node in dag.topological_op_nodes():
            if (start_time := node_start_time.get(node)) is None:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )
            if isinstance(node.op, XplusGate):
                if node.op.label == 'qutrit_toffoli_begin':
                    toffoli_begin_time = start_time
                elif node.op.label == 'qutrit_toffoli_refocusing':
                    refocusing_x12_time = start_time
                elif node.op.label == 'qutrit_toffoli_refocusing':
                    toffoli_end_time = start_time
            elif isinstance(node.op, QutritQubitCXGate):
                qutrit_qubit_cx_time = start_time
            elif dag.find_bit(node.qargs[0]).index == self.qubits[0]:
                if first_c1_node is None:
                    first_c1_node = node
                else:
                    last_c1_node = node

        for node in dag.topological_op_nodes():
            if node is first_c1_node:
                qubits = tuple(dag.find_bit(q).index for q in node.qargs)
                subdag = DAGCircuit()
                qreg = QuantumRegister(2)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(Gate('c1_dd_1', 1, []), [qreg[0]])
                subdag.apply_operation_back(XGate(), [qreg[0]])
                subdag.apply_operation_back(XGate(), [qreg[0]])
                subdag.apply_operation_back(node.op, qreg)
                dag.substitute_node_with_dag(node, subdag)
                duration = refocusing_x12_time - toffoli_begin_time
                sched = self.calibrations.get_schedule('dd', qubits[0],
                                                       assign_params={'duration': duration})
                dag.add_calibration('c1_dd_1', qubits[0], sched)

            elif node is last_c1_node:
                subdag = DAGCircuit()
                qreg = QuantumRegister(2)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(node.op, qreg)
                subdag.apply_operation_back(XGate(), [qreg[0]])
                subdag.apply_operation_back(XGate(), [qreg[0]])
                dag.substitute_node_with_dag(node, subdag)

            elif isinstance(node.op, QutritQubitCXGate):
                subdag = DAGCircuit()
                qreg = QuantumRegister(2)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(Gate('t_dd_1', 1, []), [qreg[1]])
                subdag.apply_operation_back(Gate('t_dd_2', 1, []), [qreg[1]])
                subdag.apply_operation_back(node.op, qreg)
                subdag.apply_operation_back(Gate('t_dd_3', 1, []), [qreg[1]])
                dag.substitute_node_with_dag(node, subdag)

                qubits = tuple(dag.find_bit(q).index for q in node.qargs)
                sched = get_qutrit_qubit_composite_gate(node.op.name + '_dd', qubits,
                                                        self.calibrations,
                                                        target=self.target)
                dag.add_calibration(node.op.name, qubits, sched)
                cx_duration = sched.duration

                durations = [
                    refocusing_x12_time - toffoli_begin_time,
                    qutrit_qubit_cx_time - refocusing_x12_time,
                    ((toffoli_end_time + self.instruction_durations.get('x12', qubits[0]))
                      - (qutrit_qubit_cx_time + cx_duration))
                ]
                for idx, duration in zip(range(1, 4), durations):
                    sched = self.calibrations.get_schedule('dd', qubits[1],
                                                           assign_params={'duration': duration})
                    dag.add_calibration(f't_dd_{idx}', qubits[1], sched)

        return dag
