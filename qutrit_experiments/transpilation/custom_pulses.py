"""Transpilation pass to convert calibrations with custom pulses to waveform-based ones."""
from collections import defaultdict
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass

from ..util.transforms import symbolic_pulse_to_waveform

class ConvertCustomPulses(TransformationPass):
    """Check for calibrations containing custom pulses and remove them if not used."""
    def __init__(self, remove_unused: bool):
        super().__init__()
        self.remove_unused = remove_unused

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Calibrations used in the circuit
        calib_keys = defaultdict(set)
        for node in dag.topological_op_nodes():
            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            calib_keys[node.op.name].add((qubits, tuple(node.op.params)))

        unused_gates = []
        for gate_name, instance_map in dag.calibrations.items():
            used_keys = calib_keys[gate_name]
            for instance_key, sched in list(instance_map.items()):
                if self.remove_unused and instance_key not in used_keys:
                    instance_map.pop(instance_key)
                else:
                    symbolic_pulse_to_waveform(sched)

            if len(instance_map) == 0:
                unused_gates.append(gate_name)

        if unused_gates:
            calibs = dag.calibrations
            dag.calibrations = {key: value
                                for key, value in calibs.items() if key not in unused_gates}

        return dag
