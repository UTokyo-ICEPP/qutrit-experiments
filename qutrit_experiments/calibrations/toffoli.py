"""Functions to define calibrations for the Toffoli gate."""

from collections.abc import Sequence
import logging
from typing import Optional
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations, ParameterValue

# Temporary patch for qiskit-experiments 0.5.1
from ..util.update_schedule_dependency import update_add_schedule
from .qubit import add_dd, add_ecr, add_x, set_dd_default, set_ecr_default, set_x_default

LOG = logging.getLogger(__name__)


def make_toffoli_calibrations(
    backend: Backend,
    calibrations: Calibrations,
    set_defaults: bool = True,
    qubits: Optional[Sequence[int]] = None
) -> Calibrations:
    """Define parameters and schedules for qutrit-qubit CX gate.

    Args:
        backend: Backend to use.
        calibrations: Calibrations object. If not given, a new instance is created from the backend.
        set_defaults: If True (default), sets the default values of the parameters.
        qubits: Qubits to set the default parameter values on. If not given, all qubits in the
            backend will be used.

    Returns:
        The passed or newly created Calibrations instance.
    """
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

    calibrations._register_parameter(Parameter('delta_cz'), ())
    calibrations._register_parameter(Parameter('delta_ccz'), ())

    if set_defaults:
        set_dd_default(backend, calibrations, qubits)
        if 'ecr' not in backend.basis_gates:
            set_x_default(backend, calibrations, qubits)
            set_ecr_default(backend, calibrations, qubits)

    return calibrations


def set_cz_ccz_default(
    backend: Backend,  # pylint: disable=unused-argument
    calibrations: Calibrations,
    qubits: Sequence[int]
) -> None:
    """Set the default values for local phase corrections of cz and ccz.

    Args:
        backend: Backend from which to retrieve the reference parameter values.
        calibrations: Calibrations object to define the schedules in.
        qubits: Qubits to set the parameters for. If not given, all qubits in the backend are used.
    """
    calibrations.add_parameter_value(ParameterValue(0.), 'delta_cz', qubits)
    calibrations.add_parameter_value(ParameterValue(0.), 'delta_ccz', qubits)
