"""Custom calibration updaters."""
from qiskit_experiments.calibration_management.update_library import Frequency


class EFFrequencyUpdater(Frequency):
    """Updater for EF frequency."""
    __fit_parameter__ = 'f12'
