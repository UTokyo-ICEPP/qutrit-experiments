"""Functions and transpiler passes to transpile qutrit circuits."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.providers import Backend
from qiskit.transpiler import (AnalysisPass, InstructionDurations, PassManager, Target,
                               TransformationPass, TranspilerError)

from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations

from ..gates import (QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES, QutritGate, RZ12Gate, SetF12Gate,
                     SX12Gate, X12Gate)

twopi = 2. * np.pi


def make_instruction_durations(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> InstructionDurations:
    """Construct an InstructionDurations object including qutrit gate durations."""
    if qubits is None:
        qubits = set(range(backend.num_qubits)) - set(backend.properties().faulty_qubits())

    instruction_durations = InstructionDurations(backend.instruction_durations, dt=backend.dt)
    for inst in QUTRIT_PULSE_GATES:
        durations = [(inst.gate_name, qubit,
                      calibrations.get_schedule(inst.gate_name, qubit).duration)
                     for qubit in qubits]
        instruction_durations.update(durations)
    for inst in QUTRIT_VIRTUAL_GATES:
        instruction_durations.update([(inst.gate_name, qubit, 0) for qubit in qubits])
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

    def contains_qutrit_gate(property_set):
        return property_set['contains_qutrit_gate']

    pm = PassManager()
    pm.append(ContainsQutritInstruction())
    scheduling = ALAPScheduleAnalysis(instruction_durations)
    add_cal = AddQutritCalibrations(backend.target)
    add_cal.calibrations = calibrations # See the comment in the class for why we do this
    pm.append([scheduling, add_cal], condition=contains_qutrit_gate)
    return pm.run(circuits)


class ContainsQutritInstruction(AnalysisPass):
    """Search the DAG circuit for qutrit gates."""
    def run(self, dag: DAGCircuit):
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritGate):
                self.property_set['contains_qutrit_gate'] = True
                break
        else:
            self.property_set['contains_qutrit_gate'] = False


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

        Notes about Rz and ShiftPhase
        - In Qiskit, RZGate(phi) is scheduled as ShiftPhase(-phi), which actually effects a physical
          Rz(-phi) (in the qubit space) on the backend.
        - For each qutrit gate at time t0, we need to fast-forward the pulse phase to
          omega12*t0 + qutrit_phase, where qutrit_phase is the accumulated phase shift from Rz,
          Rz12, and AC Stark corrections on the X12/SX12 pulse. Since the default phase is
          omega01*t0 + qubit_phase, we apply
          ShiftPhase((omega12-omega01)*t0 + qutrit_phase - qubit_phase)
          on each invocation of X12 and SX12 gates. As noted above, this is implemented with
          sign-inverted RZGates.
        - The correspondence of shifting the phases of X/SX/X12/SX12 pulses to physical effects
          has a global-phase ambiguity. For a phase shift of phi, any operator Q(phi) that induces
          a phase difference of phi between the upper and lower levels is a valid representation.
          X/SX/X12/SX12 at a shifted phase phi corresponds to Q(-phi) X/SX/X12/SX12 Q(phi). A
          particularly convenient representation for qutrits is
          Q(phi) = P0(-phi) = diag(exp(-iphi), 1, 1) for X/SX phase shifts and
          Q(phi) = P2(phi) = diag(1, 1, exp(iphi)) for X12/SX12. With this representation, for
          example a physical Rz(phi) gate is decomposed to
          Rz(phi) ~ diag(exp(-iphi), 1, exp(-iphi/2)) = P0(-phi)P2(-phi/2),
          which means that Rz(phi) requires a phase shift of +phi for the successive X/SX and -phi/2
          for X12/SX12. A physical Rz12(phi) is
          Rz12(phi) ~ diag(exp(iphi/2), 1, exp(iphi)) = P0(phi/2)P2(phi)
          corresponding to phase shifts of -phi/2 for X/SX and of +phi for X12/SX12.
        - Because of the sign confusion in Qiskit, however, the logical circuits are executed as
          their "mirror images" about the XZ plane of the Bloch sphere. Therefore, when we encounter
          an RZGate(phi) in the circuit, we do not try invert the phase shift but rather count it
          as P0(phi)P2(phi/2), i.e. count it as a qubit and qutrit phase shifts of -phi and +phi/2.
        """
        node_start_time = self.property_set['node_start_time']

        modulation_frequencies = {}
        qubit_phase_offsets = defaultdict(float)
        qutrit_phase_offsets = defaultdict(float)

        for node in list(dag.topological_op_nodes()):
            if node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )
            if not isinstance(node.op, (RZGate, SXGate, XGate, QutritGate)):
                continue

            qubit = dag.find_bit(node.qargs[0]).index

            if isinstance(node.op, SetF12Gate):
                if qubit in modulation_frequencies:
                    raise TranspilerError('Operation set_f12 must appear before any x12, sx12, and'
                                          ' set_f12 gates in the circuit.')
                modulation_frequencies[qubit] = (
                    node.op.params[0] - self.target.qubit_properties[qubit].frequency
                ) * self.target.dt
                dag.remove_op_node(node)
            elif isinstance(node.op, RZGate):
                # Qiskit convention
                # Rz(phi) = [ShiftPhase(-phi, qubit), ShiftPhase(0.5phi, qutrit)]
                phi = node.op.params[0]
                qubit_phase_offsets[qubit] -= phi
                qutrit_phase_offsets[qubit] += 0.5 * phi
            elif isinstance(node.op, RZ12Gate):
                # Qiskit convention
                # Rz12(phi) = [ShiftPhase(-phi, qutrit), ShiftPhase(0.5phi, qubit)]
                phi = node.op.params[0]
                qutrit_phase_offsets[qubit] -= phi
                qubit_phase_offsets[qubit] += 0.5 * phi
                # This Rz will not be processed by this pass and translates simply to
                # ShiftPhase(0.5phi, qubit)
                dag.substitute_node(node, RZGate(-0.5 * phi), inplace=True)
            elif isinstance(node.op, (XGate, SXGate)):
                # X/SX = [Play(gate_pulse, qubit), ShiftPhase(0.5delta, qutrit)]
                # See the docstring of add_x12_sx12()
                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)
                qutrit_phase_offsets[qubit] += 0.5 * delta
            else:
                if (mod_freq := modulation_frequencies.get(qubit)) is None:
                    mod_freq = (self.calibrations.get_parameter_value('f12', qubit)
                                - self.target.qubit_properties[qubit].frequency) * self.target.dt
                    modulation_frequencies[qubit] = mod_freq

                angle = qutrit_phase_offsets[qubit] - qubit_phase_offsets[qubit]
                angle += node_start_time[node] * twopi * mod_freq

                if isinstance(node.op, (X12Gate, SX12Gate)):
                    if ((qubit,), ()) not in dag.calibrations.get(node.op.name, {}):
                        assign_params = {'freq': mod_freq}
                        sched = self.calibrations.get_schedule(node.op.name, qubit,
                                                               assign_params=assign_params)
                        dag.add_calibration(node.op.name, (qubit,), sched)

                    # X12/SX12 = [Play(qutrit), ShiftPhase(-0.5delta, qubit)]
                    delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)
                    qubit_phase_offsets[qubit] -= 0.5 * delta
                else:
                    delta = 0.

                subdag = DAGCircuit()
                subdag.add_qreg((qreg := QuantumRegister(1)))
                # Qiskit convention: this is ShiftPhase(angle)
                subdag.apply_operation_back(RZGate(-angle), [qreg[0]])
                subdag.apply_operation_back(node.op, [qreg[0]])
                # ShiftPhase(-angle - 0.5delta)
                subdag.apply_operation_back(RZGate(angle + 0.5 * delta), [qreg[0]])
                subst_map = dag.substitute_node_with_dag(node, subdag)
                # Update the node_start_time map. InstructionDurations passed to the scheduling
                # pass must be constructed using the same calibrations object and therefore the
                # node duration must be consistent with sched.duration.
                start_time = node_start_time.pop(node)
                cal_key = ((qubit,), tuple(node.op.params))
                op_duration = dag.calibrations[node.op.name][cal_key].duration
                op_nodes = tuple(subdag.topological_op_nodes())
                node_start_time[subst_map[op_nodes[0]._node_id]] = start_time
                node_start_time[subst_map[op_nodes[1]._node_id]] = start_time
                node_start_time[subst_map[op_nodes[2]._node_id]] = start_time + op_duration

        return dag
