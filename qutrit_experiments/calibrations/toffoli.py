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

def make_toffoli_calibrations(
    backend: Backend,
    calibrations: Calibrations,
    qubits: tuple[int, int, int]
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    add_dd(backend, calibrations, qubits=(qubits[0], qubits[2]))
    add_qutrit_qubit_cx_dd(backend, calibrations, qubits)

    return calibrations


def add_dd(
    backend: Backend,
    calibrations: Calibrations,
    qubits: tuple[int, int]
) -> None:
    """Add templates for DD sequences (X-delay-X)."""
    inst_map = backend.defaults().instruction_schedule_map

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
    with pulse.build(name='dd') as sched:
        pulse.play(dd_pulse, pulse.DriveChannel(Parameter('ch0')))
    calibrations.add_schedule(sched, num_qubits=1)

    for qubit in qubits:
        x_sched = inst_map.get('x', qubit)
        x_pulse = x_sched.instructions[0][1].pulse
        x_duration = x_sched.duration
        pvalues = [
            ('duration', x_duration * 2),
            ('pulse_duration', x_pulse.duration),
            ('amp', x_pulse.amp),
            ('sigma', x_pulse.sigma),
            ('beta', x_pulse.beta)
        ]
        for pname, value in pvalues:
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubit],
                                             schedule=sched.name)


def add_qutrit_qubit_cx_dd(
    backend: Backend,
    calibrations: Calibrations,
    qubits: tuple[int, int, int]
) -> None:
    """Add the qutrit-qubit CX schedule including DD on the c1 qubit. To be called once all
    calibrations for the non-DD CX is settled."""
    inst_map = backend.defaults().instruction_schedule_map

    # Import the X parameter values from inst_map
    if ParameterKey('duration', (qubits[0],), 'x') not in calibrations._params:
        params = inst_map.get('x', qubits[0]).instructions[0][1].pulse.parameters
        for pname, value in params.items():
            calibrations.add_parameter_value(ParameterValue(value), pname, qubits=[qubits[0]],
                                             schedule='x')

    # X/X12 on qubit 1 with DD on qubit 2
    def dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q0')

    cr_duration = calibrations.get_schedule('cr', qubits[1:]).duration
    c1_x_sched = calibrations.get_schedule('x', qubits[:1])
    x_duration = c1_x_sched.duration

    if cr_duration >= 2 * x_duration:
        c1_cr_dd = calibrations.get_schedule('dd', qubits[0],
                                             assign_params={'duration': cr_duration})
        sched = ScheduleBlock.initialize_from(c1_cr_dd, name='c1_cr_dd')
        for block in c1_cr_dd.blocks:
            sched.append(block)
        calibrations.add_schedule(sched, qubits=qubits[0])

        # CX type X (CR > 2X)
        with pulse.build(name='qutrit_qubit_cx_rcr2_dd', default_alignment='sequential') as sched:
            for _ in range(3):
                dd()
                pulse.reference('c1_cr_dd', 'q0')
                dd()
                pulse.reference('c1_cr_dd', 'q0')
            dd()
        calibrations.add_schedule(sched, qubits=qubits[0])

        # CX type X12
        with pulse.build(name='qutrit_qubit_cx_rcr0_dd', default_alignment='sequential') as sched:
            for _ in range(3):
                pulse.reference('c1_cr_dd', 'q0')
                dd()
                pulse.reference('c1_cr_dd', 'q0')
                dd()
            dd()
        calibrations.add_schedule(sched, qubits=qubits[0])
    else:
        raise CalibrationError('CR shorter than 2xX is not supported.')
