"""Transpiler passes to transpile qutrit circuits."""
from collections import defaultdict
import logging
import numpy as np
from qiskit.circuit.library import RZGate, SXGate, XGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import AnalysisPass, Target, TransformationPass, TranspilerError
from qiskit_experiments.calibration_management import Calibrations

from ..calibrations import get_qutrit_freq_shift
from ..constants import LO_SIGN
from ..gates import (CrossResonanceGate, QutritGate, QutritQubitCXGate, RCRGate, RZ12Gate,
                     SetF12Gate, SX12Gate, X12Gate)
from .util import insert_rz

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


class ContainsQutritInstruction(AnalysisPass):
    """Search the DAG circuit for qutrit gates."""
    def run(self, dag: DAGCircuit):
        self.property_set['contains_qutrit_gate'] = False
        for node in dag.topological_op_nodes():
            if isinstance(node.op, QutritGate):
                self.property_set['contains_qutrit_gate'] = True
                break


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

        freq_diffs = {}
        cumul_angle_ge = defaultdict(float) # Angle in the g-e (|0>-|1>) sphere
        cumul_angle_ef = defaultdict(float) # Angle in the e-f (|1>-|2>) sphere

        for node in list(dag.topological_op_nodes()):
            if node not in node_start_time:
                raise TranspilerError(
                    f"Operation {repr(node)} is likely added after the circuit is scheduled. "
                    "Schedule the circuit again if you transformed it."
                )

            qubits = tuple(dag.find_bit(q).index for q in node.qargs)
            logger.debug('%s[%d]', node.op.name, qubits[0])

            if isinstance(node.op, SetF12Gate):
                if qubits[0] in freq_diffs:
                    raise TranspilerError('Operation set_f12 must appear before any x12, sx12, and'
                                          ' set_f12 gates in the circuit.')
                qubit_props = self.target.qubit_properties[qubits[0]]
                freq_diffs[qubits[0]] = node.op.params[0] - qubit_props.frequency
                dag.remove_op_node(node)

            elif isinstance(node.op, RZGate):
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

            elif isinstance(node.op, (XGate, SXGate)):
                # Corrections for geometric & Stark phases
                # X = P2(delta/2 - pi/2) U_x(pi)
                # SX = P2(delta/2 - pi/4) U_x(pi/2)
                # See the docstring of add_x12_sx12()
                # P2(phi) is equivalent to BlochRot[ef](phi)
                geom_phase = np.pi / 2. if isinstance(node.op, XGate) else np.pi / 4.
                delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubits)
                logger.debug('%s[%d] Geometric phase %f, AC Stark correction %f',
                             node.op.name, qubits[0], geom_phase, delta / 2.)
                offset = delta / 2. - geom_phase
                cumul_angle_ef[qubits[0]] += offset
                logger.debug('%s[%d] Phase[ef] += %f', node.op.name, qubits[0], offset)

            elif isinstance(node.op, QutritGate):
                pre_angles = []
                post_angles = []
                for qubit, is_qutrit in zip(qubits, node.op.qutrit):
                    if not is_qutrit:
                        pre_angles.append(0.)
                        post_angles.append(0.)
                        continue

                    # Do we know the f12 for this qubit?
                    if (freq_diff := freq_diffs.get(qubit)) is None:
                        freq_diff = get_qutrit_freq_shift(qubit, self.target, self.calibrations)
                        freq_diffs[qubit] = freq_diff
                        logger.debug('%s[%d] EF modulation frequency %f', node.op.name, qubit,
                                    freq_diff)

                    # Phase of the EF frame relative to the GE frame
                    ef_lo_phase = LO_SIGN * node_start_time[node] * twopi * freq_diff * self.target.dt
                    # Change of frame - Bloch rotation from 0 to the EF frame angle
                    # Because we share the same channel for GE and EF drives, GE angle must be offset
                    pre_angle = ef_lo_phase + cumul_angle_ef[qubit] - cumul_angle_ge[qubit]
                    post_angle = -pre_angle

                    if isinstance(node.op, (X12Gate, SX12Gate)):
                        # Corrections for geometric & Stark phases
                        # X12 = P0(delta/2 - pi/2) U_xi(pi)
                        # SX12 = P0(delta/2 - pi/4) U_xi(pi/2)
                        # P0(phi) is equivalent to BlochRot[ge](-phi)
                        geom_phase = np.pi / 2. if isinstance(node.op, X12Gate) else np.pi / 4.
                        delta = self.calibrations.get_parameter_value(f'{node.op.name}stark', qubit)
                        logger.debug('%s[%d] Geometric phase %f, AC Stark correction %f',
                                    node.op.name, qubit, geom_phase, delta / 2.)
                        offset = -(delta / 2. - geom_phase)
                        cumul_angle_ge[qubit] += offset
                        logger.debug('%s[%d] Phase[ge] += %f', node.op.name, qubit, offset)
                        post_angle += offset

                    pre_angles.append(pre_angle)
                    post_angles.append(post_angle)

                calib_key = (qubits, tuple(node.op.params))
                if (calibration := dag.calibrations.get(node.op.name, {}).get(calib_key)) is None:
                    if isinstance(node.op, (X12Gate, SX12Gate)):
                        assign_params = {'freq': freq_diff}
                    elif isinstance(node.op, CrossResonanceGate):
                        assign_params = None
                    elif isinstance(node.op, (RCRGate, QutritQubitCXGate)):
                        assign_params = {('freq', qubits[:1], 'x12'): freq_diff}
                    else:
                        raise TranspilerError(f'Missing calibration for {node.op.name}'
                                              f' ({calib_key})')
                    calibration = self.calibrations.get_schedule(node.op.name, qubits,
                                                                 assign_params=assign_params)
                    dag.add_calibration(node.op.name, qubits, calibration)
                    logger.debug('%s%s Adding calibration', node.op.name, qubits)

                insert_rz(dag, node, pre_angles=pre_angles, post_angles=post_angles,
                          node_start_time=node_start_time, op_duration=calibration.duration)


        return dag
