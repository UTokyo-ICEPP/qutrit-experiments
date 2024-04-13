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
from .util import get_default_ecr_schedule, get_operational_qubits
from .qubit import add_x, set_x_default

logger = logging.getLogger(__name__)

def make_toffoli_calibrations(
    backend: Backend,
    calibrations: Calibrations,
    set_defaults: bool = True,
    qubits: Optional[Sequence[int]] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate."""
    if calibrations is None:
        calibrations = Calibrations.from_backend(backend)
    if type(calibrations.add_schedule).__name__ == 'method':
        update_add_schedule(calibrations)

    if not calibrations.has_template('x'):
        add_x(calibrations)
    add_dd(calibrations)
    add_qutrit_qubit_cx_dd(calibrations)

    if set_defaults:
        set_toffoli_parameters(backend, calibrations, qubits)

    return calibrations


def set_toffoli_parameters(
    backend: Backend,
    calibrations: Calibrations,
    qubits: tuple[int, int, int]
):
    set_x_default(backend, calibrations, qubits=qubits)
    set_dd_default(backend, calibrations, qubits=qubits)
    # Toffoli schedule depends on the calibrated parameters specific to qubits
    instantiate_qutrit_qubit_cx_dd(calibrations, qubits)


def add_dd(
    calibrations: Calibrations
) -> None:
    """Add templates for DD sequences (X-delay-X)."""
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


def set_dd_default(
    backend: Backend,
    calibrations: Calibrations,
    qubits: Optional[Sequence[int]] = None
) -> None:
    """Copy the backend defaults X parameter values into DoubleDrag."""
    inst_map = backend.defaults().instruction_schedule_map
    operational_qubits = get_operational_qubits(backend, qubits=qubits)
    for qubit in operational_qubits:
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
                                             schedule='dd')


def add_qutrit_qubit_cx_dd(
    calibrations: Calibrations
):
    """Define the DD sequence for the c1 qubit."""
    # X/X12 on qubit 1 with DD on qubit 2
    def dd():
        with pulse.align_left():
            pulse.reference('x', 'q0')
            pulse.reference('x', 'q0')

    # CX type X (CR > 2X)
    with pulse.build(name='qutrit_qubit_cx_rcr2_dd', default_alignment='sequential') as sched:
        for _ in range(3):
            dd()
            pulse.reference('c1_cr_dd', 'q0')
            dd()
            pulse.reference('c1_cr_dd', 'q0')
        dd()
    calibrations.add_schedule(sched, num_qubits=1)

    # CX type X12
    with pulse.build(name='qutrit_qubit_cx_rcr0_dd', default_alignment='sequential') as sched:
        for _ in range(3):
            pulse.reference('c1_cr_dd', 'q0')
            dd()
            pulse.reference('c1_cr_dd', 'q0')
            dd()
        dd()
    calibrations.add_schedule(sched, num_qubits=1)


def instantiate_qutrit_qubit_cx_dd(
    calibrations: Calibrations,
    qubits: tuple[int, int, int]
) -> None:
    """Add the qutrit-qubit CX schedule including DD on the c1 qubit. To be called once all
    calibrations for the non-DD CX is settled."""
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

    else:
        raise CalibrationError('CR shorter than 2xX is not supported.')
