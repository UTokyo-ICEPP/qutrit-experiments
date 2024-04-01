"""Rough X12 and SX12 pulse amplitude calibrations based on Rabi experiments."""
from collections.abc import Iterable, Sequence
from typing import Optional
import numpy as np
from qiskit.circuit import Parameter
from qiskit.providers import Backend

from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import RoughAmplitudeCal
from qiskit_experiments.library.calibration.rough_amplitude_cal import AnglesSchedules

from .rabi import EFRabi
from ...calibrations import get_qutrit_pulse_gate


class EFRoughXSXAmplitudeCal(RoughAmplitudeCal, EFRabi):
    """Calibration experiment for X12 and SX12 amplitudes."""
    __outcome__ = 'rabi_rate_12'

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: Sequence[str] = ('x12', 'sx12'),
        amplitudes: Iterable[float] = None,
        cal_parameter_name: Sequence[str] = ('amp', 'amp'),
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
    ):
        qubit = physical_qubits[0]
        schedule = get_qutrit_pulse_gate(schedule_name[0], qubit, calibrations,
                                         target=backend.target,
                                         assign_params={cal_parameter_name[0]: Parameter("amp")})

        self._validate_channels(schedule, [qubit])
        self._validate_parameters(schedule, 1)

        super(RoughAmplitudeCal, self).__init__(
            calibrations,
            physical_qubits,
            schedule=schedule,
            amplitudes=amplitudes,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

        self.experiment_options.group = group

        angles = [np.pi, np.pi / 2.]
        self.experiment_options.angles_schedules = [
            AnglesSchedules(
                target_angle=angle,
                parameter=pname,
                schedule=sname,
                previous_value=calibrations.get_parameter_value(pname, qubit, sname)
            )
            for angle, pname, sname in zip(angles, cal_parameter_name, schedule_name)
        ]
