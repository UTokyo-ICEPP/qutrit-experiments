from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import MplDrawer

from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, make_crcr_circuit

twopi = 2. * np.pi


class CycledRepeatedCRRxAmplitude(QutritQubitTomographyScan):
    """Experiment to simultaneously scan the Rx amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.1, 0.1, 6)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: RCRType,
        param_name: str = 'amp',
        angle_param_name: str = 'sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, rx_schedule, rcr_type),
                         param_name, amplitudes, angle_param_name=angle_param_name,
                         measure_preparations=measure_preparations,
                         control_states=(0,), backend=backend)
        analyses = [exp.analysis for exp in self._experiments]
        self.analysis = CycledRepeatedCRRxAmplitudeAnalysis(analyses)


class CycledRepeatedCRRxAmplitudeAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for CycledRepeatedCRRxAmplitude."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        def curve(w, a, b):
            return (np.asarray(w) * a + b + np.pi) % twopi - np.pi

        amplitudes = np.array(experiment_data.metadata['scan_values'][0])
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        x0 = unp.nominal_values(unitaries[:, 0, 0])
        x0_steps = np.diff(x0)
        imin = np.argmin(np.abs(x0_steps))
        p0_a = x0_steps[imin] / np.diff(amplitudes)[0]
        p0_b = x0[0] - p0_a * amplitudes[0]
        popt, _ = sciopt.curve_fit(curve, amplitudes, x0, p0=(p0_a, p0_b))
        rx_amp = -popt[1] / popt[0]

        analysis_results.append(AnalysisResultData(name='rx_amp', value=rx_amp))

        if self.options.plot:
            ax = figures[0].axes[0]
            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)
            ax.plot(x_interp, curve(x_interp, *popt), color=MplDrawer.DefaultColors[0],
                    label='fit')
            ax.legend()

        return analysis_results, figures


class CycledRepeatedCRRxAmplitudeCal(BaseCalibrationExperiment, CycledRepeatedCRRxAmplitude):
    """Calibration experiment for CR width and Rx amplitude"""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['amp', 'sign_angle'],
        schedule_name: str = 'offset_rx',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        cr_schedules = [calibrations.get_schedule('cr', physical_qubits)]
        assign_params = {pname: np.pi
                         for pname in ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']}
        cr_schedules.append(calibrations.get_schedule('cr', physical_qubits,
                                                      assign_params=assign_params))
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name}
        rx_schedule = calibrations.get_schedule(schedule_name, physical_qubits[1],
                                                assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            rx_schedule,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes,
            measure_preparations=measure_preparations
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        rx_amp = experiment_data.analysis_results('rx_amp', block=False).value
        sign_angle = 0.
        if rx_amp < 0.:
            rx_amp *= -1.
            sign_angle = np.pi

        for pname, value in zip(self._param_name, [rx_amp, sign_angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )
