import numpy as np
import scipy.special as scispc
from qiskit import pulse
from qiskit.providers import Backend
from qiskit_experiments.framework import BackendData


def grounded_gauss_area(sigma: float, n: float, gs_factor: bool = False) -> float:
    """Area of a truncated Gaussian with ends grounded to zero and peak normalized to unity."""
    gauss_area = np.sqrt(2. * np.pi) * sigma * scispc.erf(n / np.sqrt(2.))
    # +1/sigma follows the Qiskit definition
    pedestal = np.exp(-0.5 * ((n + 1. / sigma) ** 2))

    area = (gauss_area - 2. * n * sigma * pedestal) / (1. - pedestal)

    if gs_factor:
        # Empirical factor 1.16 to account for an undocumented difference between
        # GaussianSquare(width=0) and Gaussian / Drag at the backend
        return area / 1.16
    else:
        return area


def rabi_freq_per_amp(backend: Backend, qubit: int) -> float:
    """Estimate the Rabi frequency from a GaussianSquare pulse."""
    x_sched = backend.defaults().instruction_schedule_map.get('x', qubit)
    x_pulse = next(inst.pulse for _, inst in x_sched.instructions if isinstance(inst, pulse.Play))
    x_area = grounded_gauss_area(x_pulse.sigma, x_pulse.duration / x_pulse.sigma / 2.,
                                 gs_factor=True)
    x_area *= np.abs(x_pulse.amp) * BackendData(backend).dt

    return 0.5 / x_area