"""Functions and transpiler passes to transpile qutrit circuits."""
from collections import defaultdict
from collections.abc import Sequence
import copy
import logging
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.providers import Backend
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.transpiler import (AnalysisPass, InstructionDurations, PassManager, Target,
                               TransformationPass, TranspilerError)

from qiskit.transpiler.passes import ALAPScheduleAnalysis
from qiskit_experiments.calibration_management import Calibrations
import rustworkx as rx

from ..constants import LO_SIGN
from ..gates import (QUTRIT_PULSE_GATES, QUTRIT_VIRTUAL_GATES, QutritGate, RZ12Gate, SetF12Gate,
                     SX12Gate, X12Gate)
from .rz import ConsolidateRZAngle, InvertRZSign

logger = logging.getLogger(__name__)
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
    instruction_durations: Optional[InstructionDurations] = None,
    resolve_rz: Optional[list[str]] = None
) -> list[QuantumCircuit]:
    """Recompute the gate durations, calculate the phase shifts for all qutrit gates, and insert
    AC Stark shift corrections to qubit gates"""
    if instruction_durations is None:
        instruction_durations = make_instruction_durations(backend, calibrations)

    def contains_qutrit_gate(property_set):
        return property_set['contains_qutrit_gate']

    pm = PassManager()
    pm.append(InvertRZSign())
    pm.append(ContainsQutritInstruction())
    scheduling = ALAPScheduleAnalysis(instruction_durations)
    add_cal = AddQutritCalibrations(backend.target, backend.configuration().channels,
                                    resolve_rz=resolve_rz)
    add_cal.calibrations = calibrations # See the comment in the class for why we do this
    pm.append([scheduling, add_cal], condition=contains_qutrit_gate)
    pm.append(ConsolidateRZAngle())
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
        target: Target,
        channel_map: dict[str, dict],
        resolve_rz: Optional[Union[str, list[str]]] = None
    ):
        super().__init__()
        # The metaclass of BasePass performs _freeze_init_parameters in which each init argument
        # is compared to some object with __eq__. Calibrations have a custom __eq__ which interferes
        # with this functionality, so we need to set self.calibrations post-init.
        self.calibrations: Calibrations = None
        self.target = target
        self.channel_map = channel_map
        if resolve_rz is None:
            self.resolve_rz = set(['ef_lo', 'ef_rz'])
        elif resolve_rz == 'all':
            self.resolve_rz = set(['ef_lo', 'ef_rz', 'ge_rz'])
        else:
            self.resolve_rz = set(resolve_rz)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Assign pulse implementations of qutrit gates.

        This class perform all of the following simultaneously:
        - Track the phase shifts on the "qutrit channel" from the Rz and Rz12 gates and the AC Stark
          shift corrections from X and SX gates.
        - For each X12 or SX12 gate, instantiate a schedule with the calculated phase shift, attach
          it to the circuit as calibration, replace the node with the parametrized version, and
          insert the geometric phase and AC Stark shift correction nodes.
        - Invert the sign of Rz and Rz12 angles if LO_SIGN>0

        Notes about Rz and ShiftPhase
        - In Qiskit, RZGate(phi) is scheduled as ShiftPhase(-phi).
        - For each qutrit gate at time t0, we need to fast-forward the pulse phase to
          LO_SIGN*omega12*t0 + qutrit_phase, where qutrit_phase is the cumululated phase shift from
          Rz, Rz12, and AC Stark corrections on the X12/SX12 pulse. Since the default phase is
          LO_SIGN*omega01*t0 + qubit_phase, we apply
          ShiftPhase(LO_SIGN*(omega12-omega01)*t0 + qutrit_phase - qubit_phase)
          on each invocation of X12 and SX12 gates. As noted above, this is implemented with
          sign-inverted RZGates.
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
          RZGate(phi) is actually effecting a physical Rz(-LO_SIGN*phi) on the qubit. Furthermore,
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
        node_start_time = self.property_set['node_start_time']

        modulation_frequencies = {}
        cumul_phase_ge = defaultdict(float) # Phase of g-e (|0>-|1>) drive
        cumul_phase_ef = defaultdict(float) # Phase of e-f (|1>-|2>) drive
        ef_lo_phase = defaultdict(float) # EF LO phase tracker
        parametrized_schedules = {} # For Rz resolution

        def insert_rz(node, pre_angle=0., post_angle=0., op_duration=0):
            subdag = DAGCircuit()
            subdag.add_qreg((qreg := QuantumRegister(len(node.qargs))))
            qargs = [qreg[0]]
            if pre_angle:
                subdag.apply_operation_back(RZGate(pre_angle), qargs)
            subdag.apply_operation_back(node.op, tuple(qreg))
            if post_angle:
                subdag.apply_operation_back(RZGate(post_angle), qargs)
            subst_map = dag.substitute_node_with_dag(node, subdag)
            # Update the node_start_time map
            start_time = node_start_time.pop(node)
            op_nodes = iter(subdag.topological_op_nodes())
            if pre_angle:
                node_start_time[subst_map[next(op_nodes)._node_id]] = start_time
            new_node = subst_map[next(op_nodes)._node_id]
            node_start_time[new_node] = start_time
            if post_angle:
                node_start_time[subst_map[next(op_nodes)._node_id]] = start_time + op_duration
            return new_node

        for node in list(dag.topological_op_nodes()):
            if node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            logger.debug('%s[%d]', node.op.name, qubits[0])

            if isinstance(node.op, SetF12Gate):
                if qubits[0] in modulation_frequencies:
                    raise TranspilerError('Operation set_f12 must appear before any x12, sx12, and'
                                          ' set_f12 gates in the circuit.')
                qubit_props = self.target.qubit_properties[qubits[0]]
                modulation_frequencies[qubits[0]] = node.op.params[0] - qubit_props.frequency
                dag.remove_op_node(node)

            elif isinstance(node.op, RZGate):
                phi = node.op.params[0]
                # Rz(phi) = ShiftPhase[ge](-phi).
                # To cancel the geometric phase, we must apply ShiftPhase[ef](phi/2)
                cumul_phase_ge[qubits[0]] -= phi
                cumul_phase_ef[qubits[0]] += phi / 2.
                logger.debug('%s[%d] Phase[ge] -= %f', node.op.name, qubits[0], phi)
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[0], phi / 2.)
                if 'ge_rz' in self.resolve_rz:
                    dag.remove_op_node(node)

            elif isinstance(node.op, RZ12Gate):
                phi = node.op.params[0]
                # Rz12(phi) = ShiftPhase[ef](-phi)
                # Then the geometric phase cancellation is ShiftPhase[ge](phi/2, qubit)]
                cumul_phase_ge[qubits[0]] += phi / 2.
                cumul_phase_ef[qubits[0]] -= phi
                logger.debug('%s[%d] Phase[ge] += %f', node.op.name, qubits[0], phi / 2.)
                logger.debug('%s[%d] Phase[ef] -= %f', node.op.name, qubits[0], phi)
                if 'ge_rz' in self.resolve_rz:
                    dag.remove_op_node(node)
                else:
                    dag.substitute_node(node, RZGate(-phi / 2.), inplace=True)

            elif isinstance(node.op, (XGate, SXGate)):
                # X = P2(delta/2 - pi/2) U_x(pi)
                # SX = P2(delta/2 - pi/4) U_x(pi/2)
                # See the docstring of add_x12_sx12()
                # P2(phi) is effected by ShiftPhase[ef](LO_SIGN * phi)
                geom_phase = np.pi / 2. if isinstance(node.op, XGate) else np.pi / 4.
                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubits)
                logger.debug('%s[%d] Geometric phase %f, AC Stark correction %f',
                             node.op.name, qubits[0], geom_phase, delta / 2.)
                offset = LO_SIGN * (delta / 2. - geom_phase)
                cumul_phase_ef[qubits[0]] += offset
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[0], offset)
                if 'ge_rz' in self.resolve_rz:
                    self.substitute_node_with_paramgate(dag, node, qubits, node_start_time[node],
                                                        (cumul_phase_ge[qubits[0]],),
                                                        parametrized_schedules)

            elif isinstance(node.op, QutritGate):
                if (mod_freq := modulation_frequencies.get(qubits[0])) is None:
                    mod_freq = (self.calibrations.get_parameter_value('f12', qubits)
                                - self.target.qubit_properties[qubits[0]].frequency)
                    modulation_frequencies[qubits[0]] = mod_freq
                    logger.debug('%s[%d] EF modulation frequency %f', node.op.name, qubits[0], mod_freq)

                ef_lo_phase[qubits[0]] = LO_SIGN * node_start_time[node] * twopi * mod_freq
                resolve_rz = {'ef_rz', 'ef_lo'} & self.resolve_rz

                if (isinstance(node.op, (X12Gate, SX12Gate)) and not resolve_rz
                    and (qubits, ()) not in dag.calibrations.get(node.op.name, {})):
                    assign_params = {'freq': mod_freq}
                    sched = self.calibrations.get_schedule(node.op.name, qubits,
                                                            assign_params=assign_params)
                    dag.add_calibration(node.op.name, qubits, sched)
                    logger.debug('%s[%d] Adding calibration', node.op.name, qubits[0])

                if resolve_rz:
                    lo_angle = ef_lo_phase[qubits[0]] - cumul_phase_ge[qubits[0]]
                    rz_angle = cumul_phase_ef[qubits[0]]
                    sched_angle = 0.
                    gate_angle = 0.
                    if 'ef_rz' in resolve_rz:
                        sched_angle += rz_angle
                    else:
                        gate_angle += rz_angle
                    if 'ef_lo' in resolve_rz:
                        sched_angle += lo_angle
                    else:
                        gate_angle += lo_angle

                    nst = node_start_time[node]
                    logger.debug('%s[%d] Adding calibration for t=%d', node.op.name, qubits[0], nst)
                    self.substitute_node_with_paramgate(dag, node, qubits, nst, (sched_angle,),
                                                        parametrized_schedules,
                                                        modulation_frequency=mod_freq)
                    op_duration = dag.calibrations[node.op.name][(qubits, (nst,))].duration
                else:
                    gate_angle = (cumul_phase_ef[qubits[0]] - cumul_phase_ge[qubits[0]]
                                  + ef_lo_phase[qubits[0]])
                    op_duration = next(s for k, s in dag.calibrations[node.op.name].values()
                                       if k[0] == qubits).duration

                if gate_angle != 0.:
                    node = insert_rz(node, pre_angle=-gate_angle, post_angle=gate_angle,
                                     op_duration=op_duration)

                # Cannot check isinstance() becaue the op may be substituted with a parametric Gate
                if node.op.name in ['x12', 'sx12']:
                    # X12 = P0(delta/2 - pi/2) U_xi(pi)
                    # SX12 = P0(delta/2 - pi/4) U_xi(pi/2)
                    # P0(phi) is effected by ShiftPhase[ge](-LO_SIGN * phi)
                    geom_phase = np.pi / 2. if isinstance(node.op, X12Gate) else np.pi / 4.
                    delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubits)
                    logger.debug('%s[%d] Geometric phase %f, AC Stark correction %f',
                                 node.op.name, qubits[0], geom_phase, delta / 2.)
                    offset = -LO_SIGN * (delta / 2. - geom_phase)
                    cumul_phase_ge[qubits[0]] += offset
                    logger.debug('%s[%d] Phase[ge] += %f', node.op.name, qubits[0], offset)
                    if 'ge_rz' not in self.resolve_rz:
                        insert_rz(node, post_angle=-offset, op_duration=op_duration)

            elif isinstance(node.op, Gate) and 'ge_rz' in self.resolve_rz:
                phases = tuple(cumul_phase_ge[q] for q in qubits)
                self.substitute_node_with_paramgate(dag, node, qubits, node_start_time[node],
                                                    phases, parametrized_schedules)

        return dag

    def substitute_node_with_paramgate(
        self,
        dag: DAGCircuit,
        node: DAGOpNode,
        qubits: tuple[int, ...],
        start_time: int,
        phases: tuple[float, ...],
        parametrized_schedules: dict[tuple[str, tuple[int, ...]], tuple[ScheduleBlock, list[Parameter]]],
        modulation_frequency: Optional[float] = None
    ):
        key = (node.op.name, qubits)
        if (psched := parametrized_schedules.get(key)) is None:
            psched = self.make_parametrized_schedule(dag, node.op, qubits,
                                                     modulation_frequency=modulation_frequency)
            parametrized_schedules[key] = psched

        sched, params = psched
        dag.substitute_node(node, Gate(sched.name, len(qubits), [start_time]), inplace=True)
        dag.add_calibration(sched.name, qubits,
                            sched.assign_parameters(dict(zip(params, phases)), inplace=False),
                            [start_time])

    def make_parametrized_schedule(
        self,
        dag: DAGCircuit,
        gate: Gate,
        qubits: tuple[int, ...],
        modulation_frequency: Optional[float] = None
    ) -> ScheduleBlock:
        if gate.__class__ in QUTRIT_PULSE_GATES:
            # There are only single-qutrit gates at the moment
            angle = Parameter('angle')
            base_angle = self.calibrations.get_parameter_value('angle', qubits, gate.name)
            assign_params = {'angle': angle + base_angle, 'freq': modulation_frequency}
            return (self.calibrations.get_schedule(gate.name, qubits, assign_params=assign_params),
                    [angle])
        try:
            schedule = next(s for k, s in dag.calibrations.get(gate.name, {}).items()
                            if k[0] == qubits)
        except StopIteration:
            schedule = self.target.instruction_schedule_map().get(gate.name, qubits)

        if isinstance(schedule, Schedule):
            # Convert to ScheduleBlock
            with pulse.build(name=schedule.name) as schedule_block:
                pulse.call(schedule)
            schedule = schedule_block
        else:
            schedule = copy.deepcopy(schedule)

        angles = self.parametrize_instructions(schedule, qubits)
        return (schedule, angles)

    def parametrize_instructions(
        self,
        schedule_block: ScheduleBlock,
        qubits: Sequence[int],
        angles: Union[Sequence[Parameter], None] = None
    ):
        if angles is None:
            angles = [Parameter(f'angle{i}') for i in range(len(qubits))]

        replacements = []
        for block in schedule_block.blocks:
            if not isinstance(block, pulse.Play):
                if isinstance(block, ScheduleBlock):
                    self.parametrize_instructions(block, qubits, angles)
                continue

            if (base_angle := block.pulse._params.get('angle')) is None:
                continue

            channel_spec = self.channel_map[block.channel.name]
            if isinstance(block.channel, pulse.DriveChannel):
                iq = qubits.index(channel_spec['operates']['qubits'][0])
            elif isinstance(block.channel, pulse.ControlChannel):
                iq = qubits.index(channel_spec['operates']['qubits'][1])
            else:
                raise RuntimeError(f'Unhandled instruction channel: {block}')

            new_inst = pulse.Play(copy.deepcopy(block.pulse), block.channel, name=block.name)
            new_inst.pulse._params['angle'] = base_angle + angles[iq]
            replacements.append((block, new_inst))

        for old, new in replacements:
            schedule_block.replace(old, new)

        # Not all parameters may have been used
        return list(set(angles) & set(schedule_block.parameters))

