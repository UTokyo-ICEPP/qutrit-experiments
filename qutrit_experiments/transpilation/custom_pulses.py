"""Transpilation pass to convert calibrations with custom pulses to waveform-based ones."""
from collections import defaultdict
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
