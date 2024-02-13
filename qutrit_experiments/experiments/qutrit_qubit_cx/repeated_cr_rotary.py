from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
import scipy.optimize as sciopt
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, make_rcr_circuit


class RepeatedCRRotaryAmplitude(QutritQubitTomographyScan):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.01, 0.01, 12)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedule: ScheduleBlock,
        rcr_type: RCRType,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits, make_rcr_circuit(physical_qubits, cr_schedule, rcr_type),
                         amp_param_name, amplitudes, angle_param_name=angle_param_name,
                         measure_preparations=measure_preparations, control_states=(1,),
                         backend=backend)
        self.analysis = RepeatedCRRotaryAmplitudeAnalysis(
            [exp.analysis for exp in self._experiments]
        )


class RepeatedCRRotaryAmplitudeAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for CycledRepeatedCRWidth."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.thetax_per_amp = None
        return options
    
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        amplitudes = np.array(experiment_data.metadata['scan_values'][0])
        if (scale_p0 := self.options.thetax_per_amp) is None:
            scale_p0 = 2. / np.amax(amplitudes)
        unitaries = next(unp.nominal_values(np.squeeze(res.value)) for res in analysis_results
                         if res.name == 'unitary_parameters')
        # unitaries has shape [scan, 3] because of np.squeeze (only one control state is used)
        # Fit a tangent curve to unitaries[:, :, 1]
        def curve(x, norm, amp0, scale, offset):
            return norm * np.tan((x - amp0) * scale) + offset
        
        norm_p0 = np.diff(unitaries[[0, -1], 1]) / np.diff(np.tan(amplitudes[[0, -1]] * scale_p0))
        center = len(amplitudes) // 2
        offset_p0 = unitaries[center, 1] - norm_p0 * np.tan(amplitudes[center] * scale_p0)
        
        popt, pcov = sciopt.curve_fit(curve, amplitudes, unitaries[:, 1],
                                      p0=(norm_p0, 0., scale_p0, offset_p0))
        popt_ufloats = correlated_values(popt, pcov)
        # Necessary rotary amplitude 
        analysis_results.append(
            AnalysisResultData(name='rotary_amp', value=popt_ufloats[1])
        )
        return analysis_results, figures


class RepeatedCRRotaryAmplitudeCal(BaseCalibrationExperiment, RepeatedCRRotaryAmplitude):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['counter_amp', 'counter_sign_angle'],
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[float] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name}
        if width is not None:
            assign_params['width'] = width
        cr_schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                                assign_params=assign_params)
        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedule,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amp_param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes,
            measure_preparations=measure_preparations
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        amplitude = experiment_data.analysis_results('rotary_amp', block=False).value
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        for pname, value in zip(self._param_name, [amplitude, angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )
