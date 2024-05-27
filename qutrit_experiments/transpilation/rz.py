"""Transpiler passses dealing with Rz angles.

Notes about Rz and ShiftPhase
  - In Qiskit, RZGate(phi) is scheduled as ShiftPhase(-phi).
  - The correspondence of shifting the phases of X/SX/X12/SX12 pulses to physical effects
    depends on the LO sign and also has a global-phase ambiguity. Drive Hamiltonian H with its
    pulse phase shifted by phi (with ShiftPhase(phi)) corresponds to
    Q(-LO_SIGN*phi) H Q(LO_SIGN*phi)
    where Q(phi) is a diagonal operator that induces a phase difference of phi between the two
    levels addressed by the drive. A particularly convenient representation of Q for
    qutrits is Q(phi) = P0(-phi) = diag(exp(-iphi), 1, 1) for X/SX phase shifts and
    Q(phi) = P2(phi) = diag(1, 1, exp(iphi)) for X12/SX12. With this representation, for
    example a physical Rz(phi) gate is decomposed to
    Rz(phi) ~ diag(exp(-iphi), 1, exp(-iphi/2)) = P0(-phi)P2(-phi/2),
    which means that Rz(phi) requires a phase shift of LO_SIGN*phi for the successive X/SX and
    -LO_SIGN*phi/2 for X12/SX12.
    Shift(phi)         -> Q(LO_SIGN*phi) = P0(-LO_SIGN*phi)
    Shift(LO_SIGN*phi) <- Q(phi)         = P0(-phi)
  - Comparing the above with the first statement in this paragraph, we see that the Qiskit
    RZGate(phi) is actually effecting a Bloch Z rotation of -LO_SIGN*phi on the qubit. Furthermore,
    as of January 2024, IBM backends have LO_SIGN=+1. When we only work in the qubit space,
    have X/SX/Rz/CX as basis gates, and measure along the Z axis, this sign confusion has no
    observable effect because the entire circuits are executed as their mirror images about
    the ZX plane of the qubit Bloch spaces. However, when a third level is considered and what
    was a global phase in the qubit space has become an observable geometric phase, this is no
    longer true. We therefore need to invert the sign of the angles of all occurrences of Rz
    and Rz12 gates.
  - Sign inversion should be done centrally in InvertRZSign so that the experiments do not
    have to worry about this issue for most part (They still need to account for the LO sign
    when using phase offsets directly in some calibration schedules).
"""
from collections import defaultdict
from collections.abc import Sequence
import copy
import logging
from typing import Union
import numpy as np
from qiskit import QuantumRegister, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RZGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.pulse import InstructionScheduleMap, Schedule, ScheduleBlock, ScalableSymbolicPulse
from qiskit.transpiler import TransformationPass, TranspilerError
import rustworkx as rx

from ..gates import RZ12Gate
from ..util.transforms import schedule_to_block
from .util import insert_rz

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class InvertRZSign(TransformationPass):
    """Transpiler pass to invert the signs of all RZ and RZ12 gates."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in list(dag.topological_op_nodes()):
            if isinstance(node.op, (RZGate, RZ12Gate)):
                # Fix the sign to make the gate correspond to its intended physical operation
                node.op.params[0] *= -1.
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

        for run in rx.collect_runs(dag._multi_graph, filter_fn):  # pylint: disable=no-member
            angle = (sum(node.op.params[0] for node in run) + np.pi) % twopi - np.pi
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


class CastRZToAngle(TransformationPass):
    """Transpiler pass to cast phase shifts by Rz instructions into pulse angles.

    This pass tracks the cumulative Rz angle per qubit and converts specified gates with pulse
    schedules. Here we are dealing with "physical RZ" i.e. instructions that get scheduled as
    ShiftPhase(-phi).

    This pass invalidates node_start_time if it is set.
    """
    def __init__(
        self,
        channel_map: dict[str, dict],
        inst_map: InstructionScheduleMap,
        gates_to_convert: Union[str, list[str]] = 'all'
    ):
        super().__init__()
        if gates_to_convert == 'all':
            self._gates = None
        else:
            self._gates = set(gates_to_convert)

        self._channel_map = channel_map
        self._inst_map = inst_map
        self._sched_cache = {}  # Container for schedules cache

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        try:
            node_start_time = self.property_set.pop('node_start_time')
        except KeyError:
            node_start_time = None
        self._sched_cache = {}  # Clear the cache
        replaced_calibrations = set()

        cumul_phase = defaultdict(float)

        for node in list(dag.topological_op_nodes()):
            if node_start_time and node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            if not isinstance(node.op, Gate):
                continue

            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            logger.debug('%s[%d]', node.op.name, qubits[0])

            if isinstance(node.op, RZGate):
                cumul_phase[qubits[0]] += node.op.params[0]
                dag.remove_op_node(node)

            elif self._gates is None or node.op.name in self._gates:
                # Rz(phi) is scheduled as ShiftPhase(-phi)
                phases = tuple(-cumul_phase[q] for q in qubits)
                # Save the original op name and params before they get replaced
                op_name = node.op.name
                op_params = tuple(node.op.params)
                # Embed the phase shifts into the pulse as angle
                sched, from_calib = self.get_schedule(dag, node.op, qubits, phases)
                gate_params = tuple(node.op.params) + tuple(phases)
                dag.substitute_node(node, Gate(node.op.name, len(qubits), gate_params),
                                    inplace=True)
                dag.add_calibration(node.op.name, qubits, sched, gate_params)
                if from_calib:
                    # The parametrized gate is derived from an entry in dag.calibrations
                    # -> Need to remove the original from calibrations at the end
                    replaced_calibrations.add((op_name, qubits, op_params))

            else:
                angles = np.array([cumul_phase[qubit] for qubit in qubits])
                insert_rz(dag, node, pre_angles=angles, post_angles=-angles,
                          node_start_time=node_start_time)
                # op_duration missing; it's likely OK because this should be one of the last passes

        for qubit, phase in cumul_phase.items():
            if phase != 0.:
                node = dag.apply_operation_back(RZGate(phase), [dag.qubits[qubit]])

        for name, qubits, params in replaced_calibrations:
            # Remove the original calibrations
            dag.calibrations[name].pop((qubits, params))
            if len(dag.calibrations[name]) == 0:
                dag.calibrations = {key: value
                                    for key, value in dag.calibrations.items() if key != name}

    def get_schedule(
        self,
        dag: DAGCircuit,
        gate: Gate,
        qubits: tuple[int, ...],
        phases: tuple[float, ...]
    ) -> tuple[ScheduleBlock, bool]:
        name = gate.name
        key = (qubits, tuple(gate.params))
        try:
            schedule, angles, from_cal = self._sched_cache[name][key]
        except KeyError:
            try:
                schedule = dag.calibrations[name][key]
            except KeyError:
                from_cal = False
                schedule = self._inst_map.get(name, qubits)
                if isinstance(schedule, Schedule):
                    schedule = schedule_to_block(schedule)
                else:
                    schedule = copy.deepcopy(schedule)
            else:
                schedule = copy.deepcopy(schedule)
                from_cal = True

            angles = self.parametrize_instructions(schedule, qubits)
            # Cache the result
            self._sched_cache.setdefault(name, {})[key] = (schedule, angles, from_cal)

        assign_params = {angle: phases[qubits.index(qubit)] for qubit, angle in angles.items()}
        schedule = schedule.assign_parameters(assign_params, inplace=False)
        self.wrap_angles(schedule)
        return schedule, from_cal

    def parametrize_instructions(
        self,
        schedule_block: ScheduleBlock,
        qubits: Sequence[int],
        angles: Union[dict[int, Parameter], None] = None
    ) -> dict[int, Parameter]:
        if angles is None:
            angles = {q: Parameter(f'angle{q}') for q in qubits}

        replacements = []
        for block in schedule_block.blocks:
            if not isinstance(block, pulse.Play):
                if isinstance(block, ScheduleBlock):
                    self.parametrize_instructions(block, qubits, angles)
                continue

            if not isinstance(block.pulse, ScalableSymbolicPulse):
                raise TranspilerError(f'Pulse {block.pulse} is not a ScalableSymbolicPulse')

            channel_spec = self._channel_map[block.channel.name]
            if isinstance(block.channel, pulse.DriveChannel):
                qubit = channel_spec['operates']['qubits'][0]
            elif isinstance(block.channel, pulse.ControlChannel):
                qubit = channel_spec['operates']['qubits'][1]
            else:
                raise RuntimeError(f'Unhandled instruction channel: {block}')

            new_inst = pulse.Play(copy.deepcopy(block.pulse), block.channel, name=block.name)
            new_inst.pulse._params['angle'] = block.pulse._params['angle'] + angles[qubit]
            replacements.append((block, new_inst))

        for old, new in replacements:
            schedule_block.replace(old, new)

        # Not all parameters may have been used
        used_params = set(schedule_block.parameters)
        return {q: angles[q] for q in qubits if angles[q] in used_params}

    def wrap_angles(
        self,
        schedule_block: ScheduleBlock
    ):
        """Wrap the angles post-assignment because ParameterExpression does not support the % op."""
        replacements = []
        for block in schedule_block.blocks:
            if not isinstance(block, pulse.Play):
                if isinstance(block, ScheduleBlock):
                    self.wrap_angles(block)
                continue

            if not isinstance(block.pulse, ScalableSymbolicPulse):
                raise TranspilerError(f'Pulse {block.pulse} is not a ScalableSymbolicPulse')

            if 0. <= block.pulse.angle < twopi:
                continue

            new_inst = pulse.Play(copy.deepcopy(block.pulse), block.channel, name=block.name)
            new_inst.pulse._params['angle'] = block.pulse._params['angle'] % twopi
            replacements.append((block, new_inst))

        for old, new in replacements:
            schedule_block.replace(old, new)
