"""Functions to define calibrations for the Toffoli gate."""

from collections.abc import Sequence
import logging
from typing import Optional
import numpy as np
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.transpiler import Target
from qiskit_experiments.calibration_management import Calibrations, ParameterValue
from qiskit_experiments.calibration_management.calibration_key_types import ParameterKey
from qiskit_experiments.exceptions import CalibrationError

from ..constants import LO_SIGN
from ..gates import ParameterValueType
from ..pulse_library import DoubleDrag
# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .util import get_default_ecr_schedule, get_operational_qubits, get_qutrit_freq_shift

logger = logging.getLogger(__name__)

def add_qutrit_qubit_cx_with_dd(
    backend: Backend,
    calibrations: Calibrations,
    qubits: tuple[int, int, int]
) -> None:
    """Add the qutrit-qubit CX schedule including DD on the c1 qubit. To be called once all
    calibrations for the non-DD CX is settled."""
    inst_map = backend.defaults().instruction_schedule_map

    q_c1, q_c2, q_t = qubits

    # Import the X parameter values from inst_map
    if ParameterKey('duration', (q_c1,), 'x') not in calibrations._params:
        params = inst_map.get('x', q_c1).instructions[0][1].pulse.parameters
        for pname, value in params.items():
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[q_c1],
                                             schedule='x')

    # X/X12 on qubit 1 with DD on qubit 2
    def x_dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q1')
            pulse.reference('x', 'q2')
            pulse.reference('x', 'q2')

    def x12_dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q0')
            pulse.reference('x12', 'q1')
            pulse.reference('x', 'q2')
            pulse.reference('x', 'q2')

    cr_duration = calibrations.get_schedule('cr', qubits[1:]).duration
    c1_x_sched = calibrations.get_schedule('x', qubits[:1])
    c1_x_pulse = c1_x_sched.instructions[0][1].pulse
    x_duration = c1_x_sched.duration

    c1_drive_channel = pulse.DriveChannel(Parameter('ch0'))
    control_channel = pulse.ControlChannel(Parameter('ch1.2'))
    target_drive_channel = pulse.DriveChannel(Parameter('ch2'))

    if cr_duration >= 2 * x_duration:
        duration = Parameter('duration')
        pulse_duration = Parameter('pulse_duration')
        dd_pulse = DoubleDrag(
            duration=duration,
            pulse_duration=pulse_duration,
            amp=Parameter('amp'),
            sigma=Parameter('sigma'),
            beta=Parameter('beta'),
            interval=(duration - 2 * pulse_duration) / 2
        )
        with pulse.build(name='c1_cr_dd') as sched:
            pulse.play(dd_pulse, c1_drive_channel)
        calibrations.add_schedule(sched, num_qubits=1)

        pvalues = [
            ('duration', cr_duration),
            ('pulse_duration', c1_x_pulse.duration),
            ('amp', c1_x_pulse.amp),
            ('sigma', c1_x_pulse.sigma),
            ('beta', c1_x_pulse.beta)
        ]
        for pname, value in pvalues:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[q_c1],
                                             schedule=sched.name)

        def cr_dd():
            with pulse.align_left():
                pulse.reference('cr', 'q1', 'q2')
                pulse.reference('c1_cr_dd', 'q0')

        # CX type X (CR > 2X)
        with pulse.build(name='qutrit_qubit_cx_rcr2_dd', default_alignment='sequential') as sched:
            # [X12+DD][CR-][X+DD][CR-]
            with pulse.phase_offset(np.pi, control_channel):
                x12_dd()
                with pulse.phase_offset(np.pi, target_drive_channel):
                    cr_dd()
                x_dd()
                cr_dd()
            for _ in range(2):
                # [X12+DD][CR+][X+DD][CR+]
                x12_dd()
                cr_dd()
                x_dd()
                with pulse.phase_offset(np.pi, target_drive_channel):
                    cr_dd()
            pulse.reference('cx_geometric_phase', 'q1')
            with pulse.align_left():
                pulse.reference('cx_offset_rx', 'q2')
                pulse.reference('x', 'q0')
                pulse.reference('x', 'q0')
                pulse.reference('x', 'q1')
                pulse.reference('x', 'q1')

        calibrations.add_schedule(sched, qubits=qubits)

        # CX type X12
        with pulse.build(name='qutrit_qubit_cx_rcr0_dd', default_alignment='sequential') as sched:
            for _ in range(2):
                # [CR+][X12+DD][CR+][X+DD]
                cr_dd()
                x12_dd()
                with pulse.phase_offset(np.pi, target_drive_channel):
                    cr_dd()
                x_dd()
            # [CR-][X12+DD][CR-][X+DD]
            with pulse.phase_offset(np.pi, control_channel):
                with pulse.phase_offset(np.pi, target_drive_channel):
                    cr_dd()
                x12_dd()
                cr_dd()
                x_dd()
            pulse.reference('cx_geometric_phase', 'q0')
            with pulse.align_left():
                pulse.reference('cx_offset_rx', 'q2')
                pulse.reference('x', 'q0')
                pulse.reference('x', 'q0')
                pulse.reference('x', 'q1')
                pulse.reference('x', 'q1')

        calibrations.add_schedule(sched, qubits=qubits)
