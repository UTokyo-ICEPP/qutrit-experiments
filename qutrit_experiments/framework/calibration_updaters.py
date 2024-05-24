"""Custom calibration updaters."""
from typing import Optional, Union
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater, Frequency


class EFFrequencyUpdater(Frequency):
    """Updater for EF frequency."""
    __fit_parameter__ = 'f12'


class DeltaUpdater(BaseUpdater):
    """Updater that increments an existing calibration value."""
    @classmethod
    def update(
        cls,
        calibrations: Calibrations,
        exp_data: ExperimentData,
        parameter: str,
        schedule: Optional[Union[ScheduleBlock, str]],
        result_index: Optional[int] = -1,
        group: str = "default",
        fit_parameter: Optional[str] = None,
    ):
        fit_parameter = fit_parameter or cls.__fit_parameter__
        delta = cls.get_value(exp_data, fit_parameter, result_index)
        value = delta + calibrations.get_parameter_value(parameter,
                                                         exp_data.metadata['physical_qubits'],
                                                         schedule=schedule, group=group)
        cls.add_parameter_value(
            calibrations, exp_data, value, parameter, schedule=schedule, group=group
        )
