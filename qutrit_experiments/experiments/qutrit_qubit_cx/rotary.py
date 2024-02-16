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
from .util import RCRType, make_crcr_circuit, make_rcr_circuit


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
                         backend=backend, analysis_cls=RepeatedCRRotaryAmplitudeAnalysis)


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
        # First identify the amplitude range where thetax has not jumped
        # point closest to zero
        center = np.argmin(np.abs(unitaries[:, 0]))
        if center == len(amplitudes) - 1:
            # Pathological case - what happened?
            slope = (np.diff(unitaries[[center - 1, center], 0])
                     / np.diff(amplitudes[[center - 1, center]]))[0]
        else:
            slope = (np.diff(unitaries[[center, center + 1], 0])
                     / np.diff(amplitudes[[center, center + 1]]))[0]
        intercept = unitaries[center, 0] - slope * amplitudes[center]
        # Points where linear prediction is not off from observation by more than 1
        indices = np.nonzero(np.abs(slope * amplitudes + intercept - unitaries[:, 0]) < 1.)[0]
        amps_selected = amplitudes[indices]
        uy_selected = unitaries[indices, 1]

        # Fit a tangent curve to uy_selected
        def curve(x, norm, amp0, scale, offset):
            return norm * np.tan((x - amp0) * scale) + offset

        norm_p0 = (np.diff(uy_selected[[0, -1]])
                   / np.diff(np.tan(amps_selected[[0, -1]] * scale_p0)))[0]
        center = len(amps_selected) // 2
        offset_p0 = uy_selected[center] - norm_p0 * np.tan(amps_selected[center] * scale_p0)

        popt, pcov = sciopt.curve_fit(curve, amps_selected, uy_selected,
                                      p0=(norm_p0, 0., scale_p0, offset_p0))
        popt_ufloats = correlated_values(popt, pcov)
        # Necessary rotary amplitude
        analysis_results.append(
            AnalysisResultData(name='rotary_amp', value=popt_ufloats[1])
        )

        if self.options.plot:
            ax = figures[1].axes[0]
            x_interp = np.linspace(amps_selected[0], amps_selected[-1], 100)
            y_interp = curve(x_interp, *popt)
            ax.plot(x_interp, y_interp, label='fit')
            ax.legend()

        return analysis_results, figures


class CycledRepeatedCRRotaryAmplitude(QutritQubitTomographyScan):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.01, 0.01, 12)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: RCRType,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, None, rcr_type),
                         amp_param_name, amplitudes, angle_param_name=angle_param_name,
                         measure_preparations=measure_preparations, backend=backend)


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
        result_index = self.experiment_options.result_index
        amplitude = BaseUpdater.get_value(experiment_data, 'rotary_amp', result_index)
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        for pname, value in zip(self._param_name, [amplitude, angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )


class CycledRepeatedCRRotaryAmplitudeCal(BaseCalibrationExperiment,
                                         CycledRepeatedCRRotaryAmplitude):
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
        cr_schedules = [calibrations.get_schedule(schedule_name, physical_qubits,
                                                  assign_params=assign_params)]
        for pname in ['cr_sign_angle', 'cr_stark_sign_phase']:
            # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
            assign_params[pname] = np.pi
        assign_params[cal_parameter_name[1]] += np.pi
        cr_schedules.append(calibrations.get_schedule(schedule_name, physical_qubits,
                                                      assign_params=assign_params))

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
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
        # Pick the rotary value with the smallest chisq?
        chisq = experiment_data.analysis_results('chisq', block=False).value
        bestfit_idx = np.argmin(np.sum(chisq, axis=1))
        amplitude = self.experiment_options.parameter_values[0][bestfit_idx]
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        for pname, value in zip(self._param_name, [amplitude, angle]):
            BaseUpdater.add_parameter_value(
                    self._cals, experiment_data, value, pname, schedule=self._sched_name,
                    group=self.experiment_options.group
                )
