"""Functions to define calibrations for the Toffoli gate."""

from collections.abc import Sequence
import logging
from typing import Optional
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .qubit import add_dd, add_ecr, add_x, set_dd_default, set_ecr_default, set_x_default


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

    if not calibrations.has_template('dd_left'):
        add_dd(calibrations)
    if 'ecr' not in backend.basis_gates:
        if not calibrations.has_template('x'):
            add_x(calibrations)
        add_ecr(calibrations)

    if set_defaults:
        set_dd_default(backend, calibrations, qubits)
        if 'ecr' not in backend.basis_gates:
            set_x_default(backend, calibrations, qubits)
            set_ecr_default(backend, calibrations, qubits)

    return calibrations
