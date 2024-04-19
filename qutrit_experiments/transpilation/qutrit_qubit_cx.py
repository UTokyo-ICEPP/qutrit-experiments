import numpy as np
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import Barrier, Delay
from qiskit.circuit.library import ECRGate, RZGate, SXGate, XGate
from qiskit.transpiler import InstructionDurations, Target, TransformationPass

from ..gates import QutritQubitCXGate, QutritQubitCXType, RZ12Gate, X12Gate, XplusGate
from .util import insert_dd


class ReverseCXDecomposition(TransformationPass):
    """Decompose QutritQubitCXGate to the double reverse ECR sequence with basis cycling."""
    def __init__(self, instruction_durations: InstructionDurations):
        super().__init__()
        self.inst_durations = instruction_durations
        self.calibrations = None
        self.dummy_circuit = False

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if not isinstance(node.op, QutritQubitCXGate):
                continue

            qids = tuple(dag.find_bit(q).index for q in node.qargs)
            rcr_type = self.calibrations.get_parameter_value('rcr_type', qids)
            if rcr_type != QutritQubitCXType.REVERSE:
                  continue

            def dur(gate, *iqs):
                return self.inst_durations.get(gate, [qids[i] for i in iqs])

            subdag = DAGCircuit()
            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)
            t = 0
            subdag.apply_operation_back(XplusGate(), [qreg[0]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(SXGate(), [qreg[1]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(XGate(), [qreg[1]])
            t += max(dur('xplus', 0), dur('sx', 1) + dur('x', 1))
            subdag.apply_operation_back(ECRGate(), [qreg[1], qreg[0]])
            t += dur('ecr', 1, 0)
            subdag.apply_operation_back(XGate(), [qreg[0]])
            subdag.apply_operation_back(RZGate(np.pi / 3.), [qreg[0]]) # Cancel the geometric phase correction
            subdag.apply_operation_back(RZ12Gate(2. * np.pi / 3.), [qreg[0]]) # Cancel the geometric phase correction
            subdag.apply_operation_back(XGate(), [qreg[1]])
            t += max(dur('x', 0), dur('x', 1))
            subdag.apply_operation_back(ECRGate(), [qreg[1], qreg[0]])
            t += dur('ecr', 1, 0)
            subdag.apply_operation_back(X12Gate(), [qreg[0]]) # We need an X+ here but would like to insert a barrier in between
            subdag.apply_operation_back(RZGate(np.pi), [qreg[1]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(SXGate(), [qreg[1]])
            subdag.apply_operation_back(RZGate(np.pi / 2.), [qreg[1]])
            subdag.apply_operation_back(Barrier(2), qreg)
            subdag.apply_operation_back(XGate(), [qreg[0]])
            subdag.apply_operation_back(Delay(t - dur('xplus', 0)), [qreg[0]])
            subdag.apply_operation_back(XplusGate(label='reverse_cx_end'), [qreg[0]])

            dag.substitute_node_with_dag(node, subdag)

        return dag


class ReverseCXDynamicalDecoupling(TransformationPass):
    def __init__(self, target: Target):
        super().__init__()
        self.target = target

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']

        for last_xplus in [node for node in dag.named_nodes('xplus')
                           if node.op.label == 'reverse_cx_end']:
            delay_node = next(dag.predecessors(last_xplus))
            x_node = next(dag.predecessors(delay_node))
            barrier_node = next(dag.predecessors(x_node))
            physical_qubits = tuple(dag.find_bit(q).index for q in barrier_node.qargs)
            x_duration = self.target['x'][(physical_qubits[1],)].calibration.duration
            xplus_duration = sum(self.calibrations.get_schedule(gate, physical_qubits[0]).duration
                                 for gate in ['x', 'x12'])

            subdag = DAGCircuit()
            start_times = []

            qreg = QuantumRegister(2)
            subdag.add_qreg(qreg)

            def add_dd(start_time, duration, placement='left'):
                need_cal = insert_dd(subdag, qreg[1], start_time, duration, x_duration,
                                     self.target.pulse_alignment, start_times, placement=placement)
                if need_cal:
                    name = f'dd_{placement}'
                    if dag.calibrations.get(name, {}).get(((physical_qubits[1],), (duration,))) is None:
                        sched = self.calibrations.get_schedule(name, physical_qubits[1],
                                                               assign_params={'duration': duration})
                        dag.add_calibration(name, [physical_qubits[1]], sched, params=[duration])

            start_time = node_start_time[x_node]
            duration = node_start_time[last_xplus] - start_time
            node = subdag.apply_operation_back(Barrier(2), qreg)
            start_times.append((node, start_time))
            add_dd(start_time, duration)
            add_dd(start_time + duration, xplus_duration)

            node_start_time.pop(barrier_node)
            subst_map = dag.substitute_node_with_dag(barrier_node, subdag)
            for node, time in start_times:
                node_start_time[subst_map[node._node_id]] = time

        return dag
