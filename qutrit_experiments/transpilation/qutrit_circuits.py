"""Transpiler passes to transpile qutrit circuits."""
from collections import defaultdict
import logging
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.circuit.library import ECRGate, RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.pulse import ScheduleBlock
from qiskit.transpiler import AnalysisPass, Target, TransformationPass, TranspilerError
from qiskit_experiments.calibration_management import Calibrations

from ..calibrations import get_qutrit_freq_shift, get_qutrit_pulse_gate
from ..calibrations.qutrit_qubit_cx import get_qutrit_qubit_cx_gate
from ..constants import LO_SIGN
from ..gates import (GateType, QutritGate, QutritQubitCXGate, RZ12Gate, SetF12Gate, SX12Gate,
                     X12Gate)
from .util import insert_rz

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


class ContainsQutritInstruction(AnalysisPass):
    """Search the DAG circuit for qutrit gates."""
    def run(self, dag: DAGCircuit):
        qutrits = set()
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritGate):
                for iq, is_qutrit in zip(node.qargs, node.op.as_qutrit):
                    if is_qutrit:
                        qutrits.add(dag.find_bit(iq).index)

        self.property_set['qutrits'] = qutrits


class AddQutritCalibrations(TransformationPass):
    """Transpiler pass to give physical implementations to qutrit gates.

    All single-qubit / single-qutrit gates act on the EF or GE two-level systems and are decomposed
    to two separate actions over the corresponding Bloch spheres. This pass converts each logical
    gate into two Bloch operations which are however confusingly expressed with the same set of
    gates. In particular, Rz gates are used to implement both the GE and EF Bloch Z rotations.

    For each qutrit gate at time t0, we need to change to the qutrit frame whose phase is
    LO_SIGN*omega12*t0 + qutrit_phase, where qutrit_phase is the cumululative phase shift from
    Rz, Rz12, and AC Stark corrections on the X12/SX12 pulse. Since the default frame is at
    LO_SIGN*omega01*t0 + qubit_phase, we need a Bloch Z rotation by -qubit_phase followed by
    LO_SIGN*(omega12-omega01)*t0 + qutrit_phase on each invocation of X12 and SX12 gates.

    This pass performs the following operations:
      - Track the Bloch Z rotations in the GE and EF spaces from the Rz and Rz12 gates, geometric
        phase corrections, and the AC Stark shift corrections.
      - Track the phase of the EF frame.
      - Attach calibrations for X12 and SX12 gates.
      - Insert BlochRot(LO_SIGN*(omega12-omega01)*t + ef_angle - ge_angle) and its inverse before
        and after each qutrit gate. FrameChange is implemented with Rz.
    """
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        node_start_time = self.property_set['node_start_time']
        dag_qutrits = self.property_set['qutrits']

        freq_diffs = {}
        cumul_angle_ge = defaultdict(float) # Angle in the g-e (|0>-|1>) sphere
        cumul_angle_ef = defaultdict(float) # Angle in the e-f (|1>-|2>) sphere
        # Rz channels cache
        rz_channels = {}
        # Drive channel to qubit mapping
        channel_qutrit_map = {}

        def get_rz_channels(qubit):
            if (channels := rz_channels.get(qubit)) is None:
                sched = self.target['rz'][(qubit,)].calibration
                channels = set(inst.channel for _, inst in sched.instructions
                               if isinstance(inst, pulse.ShiftPhase))
                rz_channels[qubit] = channels
            return channels

        def insert_ef_phase_shifts(original, ef_phase_params=None, ef_phase_shifts=None):
            if ef_phase_params is None:
                ef_phase_params = defaultdict(list)
            if ef_phase_shifts is None:
                ef_phase_shifts = defaultdict(float)

            copy = ScheduleBlock.initialize_from(original)
            for block in original.blocks:
                if isinstance(block, ScheduleBlock):
                    if block.name in ['x12', 'sx12']:
                        drive_channel = block.blocks[0].channel
                        qutrit = channel_qutrit_map[drive_channel]
                        idx = len(ef_phase_params[qutrit])
                        param = Parameter(f'ef_phase_q{qutrit}_{idx}')
                        ef_phase_params[qutrit].append(param)
                        shift = ef_phase_shifts[qutrit]
                        delta = self.calibrations.get_parameter_value(f'delta_{block.name}', qutrit)
                        geom_phase = np.pi / 2. if block.name == 'x12' else np.pi / 4.
                        # Need to apply PO(delta/2 - geom) to effect the full X12 gate
                        offset = LO_SIGN * (geom_phase - delta / 2.)
                        channels = get_rz_channels(qutrit)
                        logger.debug('  Inserting placeholder %d for qubit %d', idx, qutrit)
                        copy.append(pulse.ShiftPhase(param + shift, drive_channel))
                        copy.append(block)
                        copy.append(pulse.ShiftPhase(-(param + shift) + offset, drive_channel))
                        for channel in channels:
                            if channel != drive_channel:
                                logger.debug('    Offset %.2f to channel %s', offset, channel)
                                copy.append(pulse.ShiftPhase(offset, channel))
                    else:
                        copy.append(
                            insert_ef_phase_shifts(block, ef_phase_params, ef_phase_shifts)[0]
                        )
                elif isinstance(block, pulse.ShiftPhase) and block.name is not None:
                    logger.debug('  Found phase shift labeled as %s', block.name)
                    drive_channel = block.channel
                    qutrit = channel_qutrit_map[drive_channel]
                    if block.name == 'rz':
                        copy.append(block)
                        ef_phase_shifts[qutrit] -= block.phase / 2.
                    elif block.name == 'rz12':
                        copy.append(block)
                        # block.phase is the ge phase
                        ef_phase_shifts[qutrit] -= block.phase * 2.
                    elif block.name == 'ef_phase':
                        # Skip the instruction in copy
                        ef_phase_shifts[qutrit] += block.phase
                else:
                    copy.append(block)
            return copy, ef_phase_params

        for node in list(dag.topological_op_nodes()):
            if node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            logger.debug('%s[%s]', node.op.name, qubits)

            if isinstance(node.op, SetF12Gate):
                if qubits[0] in freq_diffs:
                    raise TranspilerError('Operation set_f12 must appear before any x12, sx12, and'
                                          ' set_f12 gates in the circuit.')
                qubit_props = self.target.qubit_properties[qubits[0]]
                freq_diffs[qubits[0]] = (node.op.params[0] - qubit_props.frequency) * self.target.dt
                dag.remove_op_node(node)

            elif isinstance(node.op, RZGate) and qubits[0] in dag_qutrits:
                phi = node.op.params[0]
                # Rz(phi) = BlochRot[ge](phi) BlochRot[ef](-phi/2)
                cumul_angle_ge[qubits[0]] += phi
                cumul_angle_ef[qubits[0]] -= phi / 2.
                logger.debug('%s[%d] Phase[ge] += %f', node.op.name, qubits[0], phi)
                logger.debug('%s[%d] Phase[ef] -= %f', node.op.name, qubits[0], phi / 2.)

            elif isinstance(node.op, RZ12Gate):
                phi = node.op.params[0]
                # Rz12(phi) = BlochRot[ge](-phi/2) BlochRot[ef](phi)
                cumul_angle_ge[qubits[0]] -= phi / 2.
                cumul_angle_ef[qubits[0]] += phi
                logger.debug('%s[%d] Phase[ge] -= %f', node.op.name, qubits[0], phi / 2.)
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[0], phi)
                # This Rz should be considered as BlochRot[ge]
                dag.substitute_node(node, RZGate(-phi / 2.), inplace=True)

            elif isinstance(node.op, (XGate, SXGate)) and qubits[0] in dag_qutrits:
                # Corrections for geometric & Stark phases
                # X = P2(delta/2 - pi/2) U_x(pi)
                # SX = P2(delta/2 - pi/4) U_x(pi/2)
                # See the docstring of add_x12_sx12()
                # P2(phi) is equivalent to BlochRot[ef](phi)
                delta = self.calibrations.get_parameter_value(f'delta_{node.op.name}', qubits[0])
                geom_phase = np.pi / 2. if isinstance(node.op, XGate) else np.pi / 4.
                offset = delta / 2. - geom_phase
                cumul_angle_ef[qubits[0]] += offset
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[0], offset)

            elif isinstance(node.op, ECRGate) and qubits[1] in dag_qutrits:
                # Corrections for Stark phase
                # ECR = IP2(delta/2 * 2) U_zx(pi/4) XI U_zx(-pi/4)
                delta = self.calibrations.get_parameter_value('delta_rzx45p_rotary', qubits)
                cumul_angle_ef[qubits[1]] += delta
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[1], delta)

            elif isinstance(node.op, QutritGate):
                logger.debug('%s[%s] at %d', node.op.name, qubits, node_start_time[node])
                for qubit, is_qutrit in zip(qubits, node.op.as_qutrit):
                    # Do we know the f12 for this qutrit?
                    if is_qutrit and qubit not in freq_diffs:
                        freq_diffs[qubit] = get_qutrit_freq_shift(qubit, self.target,
                                                                  self.calibrations)
                        logger.debug('%s[%s] EF modulation frequency %f for qutrit %d', node.op.name,
                                     qubits, freq_diffs[qubit], qubit)

                if node.op.gate_type == GateType.PULSE:
                    # Find the schedule in dag.calibrations
                    calib_key = (qubits, tuple(node.op.params))
                    if (cal := dag.calibrations.get(node.op.name, {}).get(calib_key)) is None:
                        # If not found, get the schedule from calibrations if the gate is x12/sx12
                        if isinstance(node.op, (X12Gate, SX12Gate)):
                            cal = get_qutrit_pulse_gate(node.op.name, qubits[0], self.calibrations,
                                                        freq_shift=freq_diffs[qubits[0]])
                            dag.add_calibration(node.op.name, qubits, cal)
                            logger.debug('%s%s Adding calibration', node.op.name, qubits)
                        else:
                            raise TranspilerError(f'Calibration for {node.op.name} {calib_key}'
                                                  ' missing')

                    # Calculate the phase shifts for each qubit
                    pre_angles = []
                    post_angles = []
                    for qubit, is_qutrit in zip(qubits, node.op.as_qutrit):
                        if not is_qutrit:
                            pre_angles.append(0.)
                            post_angles.append(0.)
                            continue

                        # Phase of the EF frame relative to the GE frame
                        ef_lo_phase = LO_SIGN * node_start_time[node] * twopi * freq_diffs[qubit]
                        # Change of frame - Bloch rotation from 0 to the EF frame angle
                        # Because we share the same channel for GE and EF drives, GE angle must be offset
                        pre_angles.append(ef_lo_phase + cumul_angle_ef[qubit]
                                          - cumul_angle_ge[qubit])
                        post_angles.append(-pre_angles[-1])

                        if isinstance(node.op, (X12Gate, SX12Gate)):
                            delta = self.calibrations.get_parameter_value(f'delta_{node.op.name}',
                                                                          qubit)
                            geom_phase = np.pi / 2. if isinstance(node.op, X12Gate) else np.pi / 4.
                            offset = geom_phase - delta / 2.
                            post_angles[-1] += offset
                            cumul_angle_ge[qubit] += offset
                            logger.debug('%s[%d] Phase[ge] += %f', node.op.name, qubit, offset)

                    logger.debug('%s[%s] inserting Rz pre=%s post=%s', node.op.name, qubits,
                                 pre_angles, post_angles)
                    # Phase shifts are realized with Rz gates
                    insert_rz(dag, node, pre_angles=pre_angles, post_angles=post_angles,
                              node_start_time=node_start_time, op_duration=cal.duration)

                elif node.op.gate_type == GateType.COMPOSITE:
                    # Find the schedule in dag.calibrations
                    calib_key = (qubits, tuple(node.op.params))
                    if (cal := dag.calibrations.get(node.op.name, {}).get(calib_key)) is None:
                        # If not found, get the schedule from calibrations if the gate is cx
                        if isinstance(node.op, QutritQubitCXGate):
                            cal = get_qutrit_qubit_cx_gate(qubits, self.calibrations,
                                                           freq_shift=freq_diffs[qubits[0]])
                            dag.add_calibration(node.op.name, qubits, cal, node.op.params)
                            logger.debug('%s%s Adding calibration', node.op.name, qubits)
                        else:
                            raise TranspilerError(f'Calibration for {node.op.name} {calib_key}'
                                                  ' missing')

                    # Make sure all qubit drive channels are known
                    for qubit, is_qutrit in zip(qubits, node.op.as_qutrit):
                        if is_qutrit and qubit not in channel_qutrit_map.values():
                            x_inst = self.target.get_calibration('x', (qubit,))
                            drive_channel = next(inst.channel for _, inst in x_inst.instructions
                                                 if isinstance(inst, pulse.Play))
                            channel_qutrit_map[drive_channel] = qubit

                    # Make a new ScheduleBlock with placeholder parameters for EF phase shifts
                    logger.debug('%s[%s] inserting EF phase shift placeholders', node.op.name,
                                 qubits)
                    cal, ef_phase_params = insert_ef_phase_shifts(cal)

                    # Assign phase shift values to the placeholders
                    start_time = node_start_time[node]
                    assign_map = {}
                    for inst_time, inst in cal.instructions:
                        if not isinstance(inst, pulse.Play):
                            continue
                        if (qutrit := channel_qutrit_map.get(inst.channel)) is None:
                            # Instruction with control channel etc.
                            continue
                        if inst.name.startswith('Xp'):
                            delta = self.calibrations.get_parameter_value('delta_x', qutrit)
                            cumul_angle_ef[qutrit] += delta / 2. - np.pi / 2.
                        elif inst.name.startswith('X90p'):
                            delta = self.calibrations.get_parameter_value('delta_sx', qutrit)
                            cumul_angle_ef[qutrit] += delta / 2. - np.pi / 4.
                        elif inst.name.startswith('Ξp') or inst.name.startswith('Ξ90p'):
                            # See comments on X12Gate
                            ef_lo_phase = (LO_SIGN * (start_time + inst_time) * twopi
                                           * freq_diffs[qutrit])
                            parameter = ef_phase_params[qutrit].pop(0)
                            assign_map[parameter] = (ef_lo_phase + cumul_angle_ef[qutrit]
                                                     - cumul_angle_ge[qutrit]) % twopi
                            logger.debug('  Assigning %.2f to %s', assign_map[parameter], parameter)
                            if inst.name.startswith('Ξp'):
                                delta = self.calibrations.get_parameter_value('delta_x12', qutrit)
                                cumul_angle_ge[qutrit] += np.pi / 2. - delta / 2.
                            else:
                                delta = self.calibrations.get_parameter_value('delta_sx12', qutrit)
                                cumul_angle_ge[qutrit] += np.pi / 4. - delta / 2.

                    assert set(len(l) for l in ef_phase_params.values()) == {0}

                    node.op.params.append(start_time)
                    cal.assign_parameters(assign_map, inplace=True)
                    dag.add_calibration(node.op.name, qubits, cal, node.op.params)

                else:
                    raise TranspilerError(f'Unhandled qutrit gate {node.op.name}')

        return dag
