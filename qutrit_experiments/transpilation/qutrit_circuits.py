from collections import defaultdict
from collections.abc import Sequence
from typing import Optional, Union
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import Backend
from qiskit.transpiler import (InstructionDurations, PassManager, Target, TransformationPass,
                               TranspilerError)
from qiskit.transpiler.passes.scheduling import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations

from ..gates import RZ12Gate, SX12Gate, X12Gate


def make_instruction_durations(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> InstructionDurations:
    """Construct an InstructionDurations object including qutrit gate durations."""
    if qubits is None:
        qubits = set(range(backend.num_qubits)) - set(backend.properties().faulty_qubits())

    instruction_durations = InstructionDurations(backend.instruction_durations, dt=backend.dt)
    for inst_name in ['x12', 'sx12']:
        durations = [(inst_name, qubit, calibrations.get_schedule(inst_name, qubit).duration)
                     for qubit in qubits]
        instruction_durations.update(durations)
    instruction_durations.update([('rz12', qubit, 0) for qubit in qubits])
    return instruction_durations


def transpile_qutrit_circuits(
    circuits: Union[QuantumCircuit, list[QuantumCircuit]],
    backend: Backend,
    calibrations: Calibrations,
    instruction_durations: Optional[InstructionDurations] = None
) -> list[QuantumCircuit]:
    """Recompute the gate durations, calculate the phase shifts for all qutrit gates, and insert
    AC Stark shift corrections to qubit gates"""
    if instruction_durations is None:
        instruction_durations = make_instruction_durations(backend, calibrations)

    pm = PassManager()
    pm.append(ALAPScheduleAnalysis(instruction_durations))
    add_cal = AddQutritCalibrations(backend.target)
    add_cal.calibrations = calibrations # See the comment in the class for why we do this
    pm.append(add_cal)
    return pm.run(circuits)


class AddQutritCalibrations(TransformationPass):
    """Transpiler pass to give physical implementations to qutrit gates."""
    def __init__(
        self,
        target: Target
    ):
        super().__init__()
        # The metaclass of BasePass performs _freeze_init_parameters in which each init argument
        # is compared to some object with __eq__. Calibrations have a custom __eq__ which interferes
        # with this functionality, so we need to set self.calibrations post-init.
        self.calibrations: Calibrations = None
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
                # Rz(phi) = [ShiftPhase(-phi, qubit_channel), ShiftPhase(0.5phi, qutrit_channel)]
                qubit_phase_offsets[qubit] -= node.op.params[0]
                qutrit_phase_offsets[qubit] += 0.5 * node.op.params[0]
            elif isinstance(node.op, RZ12Gate):
                # Rz12(phi) = [ShiftPhase(-phi, qutrit_channel), ShiftPhase(0.5phi, qubit_channel)]
                qutrit_phase_offsets[qubit] -= node.op.params[0]
                # This Rz translates simply to ShiftPhase(0.5phi, qubit_channel)
                dag.substitute_node(node, RZGate(-0.5 * node.op.params[0]), inplace=True)
                qubit_phase_offsets[qubit] += 0.5 * node.op.params[0]
            elif isinstance(node.op, (XGate, SXGate)):
                # X/SX = [Play(qubit_channel), ShiftPhase(-0.5delta, qutrit_channel)]
                # See the docstring of add_x12_sx12()
                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)
                qutrit_phase_offsets[qubit] -= 0.5 * delta
            else:
                if (mod_freq := modulation_frequencies.get(qubit)) is None:
                    mod_freq = (self.calibrations.get_parameter_value('f12', qubit)
                                - self.target.qubit_properties[qubit].frequency) * self.target.dt
                    modulation_frequencies[qubit] = mod_freq

                angle = qutrit_phase_offsets[qubit] - qubit_phase_offsets[qubit]
                angle += node_start_time[node] * mod_freq
                assign_params = {'freq': mod_freq, 'angle': angle}
                sched = self.calibrations.get_schedule(node.op.name, qubit,
                                                       assign_params=assign_params)
                dag.add_calibration(node.op.name, (qubit,), sched, [angle])

                # X12/SX12 = [Play(qutrit_channel), ShiftPhase(0.5delta, qubit_channel)]
                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)
                qubit_phase_offsets[qubit] += 0.5 * delta

                subdag = DAGCircuit()
                subdag.add_qreg((qreg := QuantumRegister(1)))
                subdag.apply_operation_back(node.op.__class__(angle), [qreg[0]])
                subdag.apply_operation_back(RZGate(-0.5 * delta), [qreg[0]])
                subst_map = dag.substitute_node_with_dag(node, subdag)
                # Update the node_start_time map. InstructionDurations passed to the scheduling
                # pass must be constructed using the same calibrations object and therefore the
                # node duration must be consistent with sched.duration.
                start_time = node_start_time.pop(node)
                op_node, rz_node = tuple(subdag.topological_op_nodes())
                node_start_time[subst_map[op_node._node_id]] = start_time
                node_start_time[subst_map[rz_node._node_id]] = start_time + sched.duration

        return dag
