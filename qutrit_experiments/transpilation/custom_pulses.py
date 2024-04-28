"""Transpilation pass to convert calibrations with custom pulses to waveform-based ones."""
from collections.abc import Sequence
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

from ..util.transforms import symbolic_pulse_to_waveform


class RemoveUnusedCalibrations(TransformationPass):
    """Remove calibrations with no corresponding gates."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        current = dag.calibrations
        used = {}
        for node in dag.topological_op_nodes():
            if (instances := current.get(node.op.name)) is None:
                continue
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            key = (qubits, tuple(node.op.params))
            if (sched := instances.get(key)) is not None:
                used.setdefault(node.op.name, {})[key] = sched

        dag.calibrations = used
        return dag


class ConvertCustomPulses(TransformationPass):
    """Convert non-standard pulses to waveforms."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for instance_map in dag.calibrations.values():
            for sched in instance_map.values():
                symbolic_pulse_to_waveform(sched)

        return dag


class AttachCalibration(TransformationPass):
    """Attach a calibration of a given name to DAG."""
    def __init__(self, gates: Sequence[str]):
        super().__init__()
        self.gates = set(gates)
        self.calibrations = None

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            if not node.op.name in self.gates:
                continue

            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            if dag.calibrations.get(node.op.name, {}).get((qubits, ())) is None:
                sched = self.calibrations.get_schedule(node.op.name, qubits)
                dag.add_calibration(node.op.name, qubits, sched)

        return dag
