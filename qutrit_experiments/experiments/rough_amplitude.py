"""Rough X12 and SX12 pulse amplitude calibrations based on Rabi experiments."""

from collections.abc import Iterable, Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import Options
from qiskit_experiments.library import RoughAmplitudeCal
from qiskit_experiments.library.calibration.rough_amplitude_cal import AnglesSchedules

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from .rabi import Rabi
from ..util.dummy_data import ef_memory, single_qubit_counts


class EFRabi(EFSpaceExperiment, Rabi):
    """Rabi experiment with initial and final X gates."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(-0.4, 0.4, 17)
        return options

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        amplitudes = self.experiment_options.amplitudes
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = np.cos(2. * np.pi * 4. * amplitudes) * 0.49 + 0.51
        num_qubits = 1

        if self._final_xgate:
            states = (0, 2)
        else:
            states = (1, 2)

        if self.run_options.meas_level == MeasLevel.KERNELED:
            return ef_memory(one_probs, shots, num_qubits,
                             meas_return=self.run_options.get('meas_return', 'avg'),
                             states=states)

        return single_qubit_counts(one_probs, shots, num_qubits)


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
        backend: Optional[Backend] = None
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name=schedule_name[0],
            amplitudes=amplitudes,
            backend=backend,
            cal_parameter_name=cal_parameter_name[0],
            target_angle=np.pi,
        )
        self._sched_name = list(schedule_name)
        self._param_name = list(cal_parameter_name)

        angles = [np.pi, np.pi / 2.]
        self.experiment_options.angles_schedules = [
            AnglesSchedules(
                target_angle=angle,
                parameter=pname,
                schedule=sname,
                previous_value=None
            )
            for angle, pname, sname in zip(angles, cal_parameter_name, schedule_name)
        ]

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> dict[str, Any]:
        """Call EFRoughXSXAmplitudeCal._metadata with experiment options group set to default."""
        current_group = self.experiment_options.group
        self.set_experiment_options(group='default')
        metadata = super()._metadata()
        metadata['cal_group'] = current_group
        self.set_experiment_options(group=current_group)

        return metadata
