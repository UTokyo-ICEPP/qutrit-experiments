from collections import defaultdict
from qiskit import QuantumRegister
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.transpiler import Target, TransformationPass, TranspilerError
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit_experiments.calibration_management import Calibrations

from ..gates import RZ12Gate, SX12Gate, X12Gate


class AddQutritCalibrations(TransformationPass):
    def __init__(
        self,
        target: Target
    ):
        super().__init__()
        # The metaclass of BasePass performs _freeze_init_parameters in which each init argument
        # is compared to some object with __eq__. Calibrations have a custom __eq__ which interferes
        # with this functionality, so we need to set self.calibrations post-init.
        self.calibrations = None
        self.target = target

    def run(self, dag: DAGCircuit):
        """Assign pulse implementations of qutrit gates.

        This class perform all of the following simultaneously:
        - Track the phase shifts on the "qutrit channel" from the Rz and Rz12 gates and the AC Stark
          shift corrections from X and SX gates.
        - For each X12 or SX12 gate, instantiate a schedule with the calculated phase shift, attach
          it to the circuit as calibration, replace the node with the parametrized version, and
          insert the AC Stark shift correction nodes.
        """
        node_start_time = self.property_set['node_start_time']

        modulation_frequencies = {}
        qubit_phase_offsets = defaultdict(float)
        qutrit_phase_offsets = defaultdict(float)

        qutrit_gates = ()

        for node in dag.topological_op_nodes():
            if node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )
            if not isinstance(node.op, (RZGate, SXGate, XGate, RZ12Gate, SX12Gate, X12Gate)):
                continue

            qubit = dag.find_bit(node.qargs[0]).index

            if isinstance(node.op, RZGate):
                qubit_phase_offsets[qubit] -= node.op.params[0]
                # shift_phase(-phi) applied to the qubit channel -> tally up +0.5phi for qutrit
                qutrit_phase_offsets[qubit] += 0.5 * node.op.params[0]
            elif isinstance(node.op, RZ12Gate):
                qutrit_phase_offsets[qubit] -= node.op.params[0]
                dag.substitute_node(node, RZGate(-0.5 * node.op.params[0]), inplace=True)
                qubit_phase_offsets[qubit] -= 0.5 * node.op.params[0]
            elif isinstance(node.op, XGate):
                delta = self.calibrations.get_parameter_value('xstark', qubit)
                qutrit_phase_offsets[qubit] -= 0.5 * delta
            elif isinstance(node.op, SXGate):
                delta = self.calibrations.get_parameter_value('sxstark', qubit)
                qutrit_phase_offsets[qubit] -= 0.5 * delta
            else:
                if (mod_freq := modulation_frequencies.get(qubit)) is None:
                    mod_freq = (self.calibrations.get_parameter_value('f12', qubit)
                                - self.target.qubit_properties[qubit].frequency) * self.target.dt
                    modulation_frequencies[qubit] = mod_freq

                phase_offset = qutrit_phase_offsets[qubit] - qubit_phase_offsets[qubit]
                phase_offset += node_start_time[node] * mod_freq

                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)

                subdag = DAGCircuit()
                subdag.add_qreg((qreg := QuantumRegister(1)))
                pulse_node = subdag.apply_operation_back(node.op.__class__(phase_offset), [qreg[0]])
                rz_node = subdag.apply_operation_back(RZGate(0.5 * delta), [qreg[0]])
                qubit_phase_offsets[qubit] += 0.5 * delta
                subst_map = dag.substitute_node_with_dag(node, subdag)

                assign_params = {'phase_offset': phase_offset}
                sched = self.calibrations.get_schedule(node.op.name, qubit,
                                                       assign_params=assign_params)
                dag.add_calibration(node.op.name, (qubit,), sched, [phase_offset])

                # Update the node_start_time map. InstructionDurations passed to the scheduling
                # pass must be constructed using the same calibrations object and therefore the
                # node duration must be consistent with sched.duration.
                start_time = node_start_time.pop(node)
                node_start_time[subst_map[pulse_node._node_id]] = start_time
                node_start_time[subst_map[rz_node._node_id]] = start_time + sched.duration

        return dag
