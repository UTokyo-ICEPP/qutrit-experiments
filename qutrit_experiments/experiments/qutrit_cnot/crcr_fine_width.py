from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

from ...common.framework_overrides import CompoundAnalysis, BatchExperiment
from ...common.util import get_cr_margin
from .rx_cr_width_scan import CXPingPong


twopi = 2. * np.pi

class QutritCXCRCRFineWidth(BatchExperiment):
    """Perform error amplification experiments for control=0 and 1."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: Optional[ScheduleBlock] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for control_state in [0, 1]:
            exp = CXPingPong(physical_qubits, control_state=control_state, schedule=schedule,
                             backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=QutritCXCRCRFineWidthAnalysis(analyses))


class QutritCXCRCRFineWidthAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        component_index = experiment_data.metadata["component_child_index"]

        d_theta = [experiment_data.child_data(idx).analysis_results('d_theta').value
                   for idx in component_index]

        analysis_results += [
            AnalysisResultData(name='crcr_phi0', value=d_theta[0]),
            AnalysisResultData(name='crcr_rabi_phase_diff', value=(d_theta[1] - d_theta[0]))
        ]

        return analysis_results, figures


class QutritCXCRCRFineWidthCal(BaseCalibrationExperiment, QutritCXCRCRFineWidth):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = ['cr', 'cr', 'rx', 'rx'],
        cal_parameter_name: List[str] = ['cr_width', 'cr_margin', 'rx_amp', 'rx_angle'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend
        )

        self.schedule = calibrations.get_schedule('qutrit_cx', physical_qubits)
        self.cx_rabi_frequency_diff = calibrations.get_parameter_value('cx_rabi_frequency_diff',
                                                                       physical_qubits,
                                                                       schedule='cx_rabi')
        self.rx_rabi_frequency = calibrations.get_parameter_value('rx_rabi_frequency',
                                                                  physical_qubits,
                                                                  schedule='cx_rabi')

    def _attach_calibrations(self, circuit: QuantumCircuit):
        super()._attach_calibrations(circuit)
        circuit.add_calibration('cx', self.physical_qubits, self.schedule)

    def _metadata(self) -> Dict[str, Any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=sname,
                group=self.experiment_options.group,
            ) for pname, sname in zip(self._param_name, self._sched_name)
        ]

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group
        result_index = self.experiment_options.result_index

        current_width = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits,
                                                       schedule=self._sched_name[0])
        rabi_phase_diff = BaseUpdater.get_value(experiment_data, 'crcr_rabi_phase_diff',
                                                index=result_index)
        new_width = current_width + rabi_phase_diff / (self.cx_rabi_frequency_diff * twopi
                                                       * self._backend_data.dt)
        new_margin = get_cr_margin(new_width, self.backend, self._cals, self.physical_qubits,
                                   self._sched_name[0])
        current_rx_amp = self._cals.get_parameter_value(self._param_name[2], self.physical_qubits,
                                                        schedule=self._sched_name[2])
        current_rx_angle = self._cals.get_parameter_value(self._param_name[3], self.physical_qubits,
                                                        schedule=self._sched_name[3])
        if current_rx_angle != 0.:
            current_rx_amp *= -1.
        crcr_offset = BaseUpdater.get_value(experiment_data, 'crcr_phi0', index=result_index)
        new_rx_amp = current_rx_amp + crcr_offset / self.rx_rabi_frequency / twopi
        new_rx_angle = 0.
        if new_rx_amp < 0.:
            new_rx_amp = abs(new_rx_amp)
            new_rx_angle = np.pi

        for pname, sname, value in zip(self._param_name, self._sched_name,
                                       [new_width, new_margin, new_rx_amp, new_rx_angle]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=sname, group=group)
