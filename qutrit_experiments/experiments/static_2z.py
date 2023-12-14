from typing import Any, Dict, List, Sequence, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations, BaseCalibrationExperiment
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import ExperimentData
from .ramsey import QutritRamseyXY


twopi = 2. * np.pi

class Static2ZCal(BaseCalibrationExperiment, QutritRamseyXY):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'static_omega_2z',
        cal_parameter_name: str = 'static_omega_2z',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        delays: Optional[List] = None,
        osc_freq: Optional[float] = None
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            2,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            delays=delays,
            osc_freq=osc_freq
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            schedule=self._sched_name,
            group=self.experiment_options.group
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        freq = BaseUpdater.get_value(experiment_data, 'freq', index=result_index)
        BaseUpdater.add_parameter_value(self._cals, experiment_data, twopi * freq, self._param_name,
                                        schedule=self._sched_name, group=group)
