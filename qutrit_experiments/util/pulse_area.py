"""Subroutines to compute the pulse areas."""
from typing import Optional
import numpy as np
import scipy.special as scispc
from qiskit import pulse
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations


def grounded_gauss_area(sigma: float, rsr: float, gs_factor: bool = False) -> float:
    """Area of a truncated Gaussian with ends grounded to zero and peak normalized to unity."""
    # pylint: disable=no-member
    gauss_area = np.sqrt(2. * np.pi) * sigma * scispc.erf(rsr / np.sqrt(2.))
    # +1/sigma follows the Qiskit definition
    pedestal = np.exp(-0.5 * ((rsr + 1. / sigma) ** 2))

    area = (gauss_area - 2. * rsr * sigma * pedestal) / (1. - pedestal)

    if gs_factor:
        # Empirical factor 1.16 to account for an undocumented difference between
        # GaussianSquare(width=0) and Gaussian / Drag at the backend
        return area / 1.16
    else:
        return area


def gs_effective_duration(
    calibrations: Calibrations,
    qubits: tuple[int, int],
    schedule: str,
    width: Optional[float] = None
) -> float:
    """Duration of the square pulse with the same area as the GS pulse."""
    sigma = calibrations.get_parameter_value('sigma', qubits, schedule)
    rsr = calibrations.get_parameter_value('rsr', qubits, schedule)
    if width is None:
        width = calibrations.get_parameter_value('width', qubits, schedule)

    return grounded_gauss_area(sigma, rsr, gs_factor=True) + width


def rabi_cycles_per_area(backend: Backend, qubit: int) -> float:
    """Estimate the Rabi rotation cycles per pulse area."""
    x_sched = backend.defaults().instruction_schedule_map.get('x', qubit)
    x_pulse = next(inst.pulse for _, inst in x_sched.instructions if isinstance(inst, pulse.Play))
    x_area = grounded_gauss_area(x_pulse.sigma, x_pulse.duration / x_pulse.sigma / 2.,
                                 gs_factor=True)
    x_area *= np.abs(x_pulse.amp)
    return 0.5 / x_area
