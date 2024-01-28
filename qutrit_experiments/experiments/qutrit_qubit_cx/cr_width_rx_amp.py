from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np
import scipy.optimize as sciopt
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import XGate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import (AnalysisResultData, BackendData, BackendTiming,
                                          ExperimentData, Options)
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...gates import X12Gate
from ..qutrit_qubit_qpt import QutritQubitQPTScan, QutritQubitQPTScanAnalysis

twopi = 2. * np.pi


def make_crcr_circuit(
    physical_qubits: Sequence[int],
    cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
    rx_schedule: Union[ScheduleBlock, None],
    rcr_type: str
) -> QuantumCircuit:
    crp_params = cr_schedules[0].parameters
    crm_params = cr_schedules[1].parameters
    if rx_schedule is not None:
        rx_params = rx_schedule.parameters

    crcr_circuit = QuantumCircuit(2)
    if rcr_type == 'x':
        # [X+]-[RCR-]
        if rx_schedule is not None:
            crcr_circuit.append(Gate('offset_rx', 1, rx_params), [1])
        crcr_circuit.append(X12Gate())
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.x(0)
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        # [X+]-[RCR+] x 2
        for _ in range(2):
            crcr_circuit.append(X12Gate())
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.x(0)
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
    else:
        # [RCR+]-[X+] x 2
        for _ in range(2):
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.append(X12Gate(), [0])
            crcr_circuit.append(Gate('crp', 2, crp_params), [0, 1])
            crcr_circuit.x(0)
        # [RCR-]-[X+]
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.append(X12Gate(), [0])
        crcr_circuit.append(Gate('crm', 2, crm_params), [0, 1])
        crcr_circuit.x(0)
        if rx_schedule is not None:
            crcr_circuit.append(Gate('offset_rx', 1, rx_params), [1])

    crcr_circuit.add_calibration('crp', physical_qubits, cr_schedules[0], crp_params)
    crcr_circuit.add_calibration('crm', physical_qubits, cr_schedules[1], crm_params)
    if rx_schedule is not None:
        crcr_circuit.add_calibration('offset_rx', (physical_qubits[1],), rx_schedule, rx_params)

    return crcr_circuit


def get_margin(
    cr_schedule: ScheduleBlock,
    width_param_name: str,
    margin_param_name: str,
    widths: Union[float, Sequence[float]],
    backend: Backend
):
    width_param = cr_schedule.get_parameters(width_param_name)[0]
    margin_param = cr_schedule.get_parameters(margin_param_name)[0]
    test_assign = {width_param: 0., margin_param: 0.}
    risefall_duration = cr_schedule.assign_parameters(test_assign, inplace=False).duration
    backend_timing = BackendTiming(backend)
    granularity = BackendData(backend).granularity
    supports = np.asarray(widths) + risefall_duration
    if (is_scalar := (widths.ndim == 0)):
        supports = [supports]

    margins = np.array([backend_timing.round_pulse(samples=support) - support
                        for support in supports])
    margins = np.where(margins < 0., margins + granularity, margins)
    if is_scalar:
        return margins[0]
    else:
        return margins


class CycledRepeatedCRWidth(QutritQubitQPTScan):
    """Experiment to simultaneously scan the CR width."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        widths = np.linspace(0., 128., 5)
        margins = np.zeros(5)
        options.parameter_values = [widths, margins, widths, margins]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: str,
        width_param_name: str = 'width',
        margin_param_name: str = 'margin',
        widths: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if widths is None:
            widths = self._default_experiment_options().parameter_values[0]

        margins = get_margin(cr_schedules[0], width_param_name, margin_param_name, widths, backend)

        # Rename the CR parameters to distinguish crp and crm
        for idx, (prefix, sched) in enumerate(zip(['crp', 'crm'], cr_schedules)):
            assign_params = {sched.get_parameters(pname)[0]: Parameter(f'{prefix}_{pname}')
                             for pname in [width_param_name, margin_param_name]}
            cr_schedules[idx] = sched.assign_parameters(assign_params, inplace=False)

        param_names = [f'{prefix}_{pname}'
                       for prefix in ['crp', 'crm']
                       for pname in [width_param_name, margin_param_name]]
        param_values = [widths, margins, widths, margins]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, None, rcr_type),
                         param_names, param_values, backend=backend)
        self.analysis = CycledRepeatedCRWidthAnalysis([exp.analysis for exp in self._experiments])


class CycledRepeatedCRWidthAnalysis(QutritQubitQPTScanAnalysis):
    """Analysis for CycledRepeatedCRWidth."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        def curve(w, a, b):
            return (np.asarray(w) * a + b + np.pi) % twopi - np.pi

        widths = next(res.value for res in analysis_results if res.name == 'crp_width')
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        popts = []
        for control_state in range(2):
            xs = unitaries[:, control_state, 0]
            x_steps = np.diff(xs)
            imin = np.argmin(np.abs(x_steps))
            p0_a = x_steps[imin] / np.diff(widths)[0]
            p0_b = xs[0] - p0_a * widths[0]
            popt, _ = sciopt.curve_fit(curve, widths, xs, p0=(p0_a, p0_b))
            popts.append(popt)
        slope = popts[1][0] - popts[0][0]
        intercept = popts[1][1] - popts[0][1]
        # Find the minimum w>0 that satisfies a*w + b = (2n+1)*π  (n ∈ Z)
        wmin = (np.pi - intercept) / slope
        while wmin < 0.:
            wmin += twopi / np.abs(slope)
        while (wtest := wmin - twopi / np.abs(slope)) > 0.:
            wmin = wtest
        rabi_phase_diff = slope * wmin + intercept

        analysis_results.extend([
            AnalysisResultData(name='cr_width', value=wmin),
            AnalysisResultData(name='cx_sign', value=np.sign(rabi_phase_diff))
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='CR width',
                ylabel=r'$\theta_1 - \theta_0$',
                ylim=(-twopi - 0.1, twopi + 0.1)
            )
            x_interp = np.linspace(widths[0], widths[-1], 100)
            plotter.set_series_data(
                'angle_diff',
                x_formatted=widths,
                y_formatted=(unitaries[:, 1, 0] - unitaries[:, 0, 0] + twopi) % (2. * twopi) - twopi,
                y_formatted_err=np.zeros_like(widths),
                x_interp=x_interp,
                y_interp=curve(x_interp, slope, intercept)
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class CycledRepeatedCRWidthCal(BaseCalibrationExperiment, CycledRepeatedCRWidth):
    """Calibration experiment for CR width and Rx amplitude"""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        rcr_type: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin'],
        schedule_name: str = 'cr',
        widths: Optional[Sequence[float]] = None,
        auto_update: bool = True
    ):
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name}
        cr_schedules = [calibrations.get_schedule(schedule_name, physical_qubits,
                                                  assign_params=assign_params)]
        for pname in ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']:
            # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
            assign_params[pname] = np.pi
        cr_schedules.append(calibrations.get_schedule(schedule_name, physical_qubits,
                                                      assign_params=assign_params))

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            rcr_type,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            width_param_name=cal_parameter_name[0],
            margin_param_name=cal_parameter_name[1],
            widths=widths
        )
        # Need to have circuits() return decomposed circuits because of how _transpiled_circuits
        # work in calibration experiments
        for qutrit_qpt in self.component_experiment():
            for qpt in qutrit_qpt.component_experiment():
                qpt.set_experiment_options(decompose_circuits=True)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        width = experiment_data.analysis_results('cr_width', block=False).value
        assign_params = {pname: Parameter(pname) for pname in self._param_name}
        sched_template = self._cals.get_schedule(self._sched_name, self.physical_qubits,
                                                 assign_params)
        margin = get_margin(sched_template, self._param_name[0], self._param_name[1], width,
                            self._backend)

        for pname, value in zip(self._param_name, [width, margin]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )


class CycledRepeatedCRRxAmplitude(QutritQubitQPTScan):
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
        rcr_type: str,
        param_name: str = 'amp',
        angle_param_name: str = 'sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, rx_schedule, rcr_type),
                         param_name, amplitudes, angle_param_name=angle_param_name,
                         backend=backend)
        analyses = [exp.analysis for exp in self._experiments]
        self.analysis = CycledRepeatedCRRxAmplitudeAnalysis(analyses)


class CycledRepeatedCRRxAmplitudeAnalysis(QutritQubitQPTScanAnalysis):
    """Analysis for CycledRepeatedCRRxAmplitude."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        def curve(w, a, b):
            return (np.asarray(w) * a + b + np.pi) % twopi - np.pi

        amplitudes = next(res.value for res in analysis_results if res.name == 'amp')
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        x0 = unitaries[:, 0, 0]
        x0_steps = np.diff(x0)
        imin = np.argmin(np.abs(x0_steps))
        p0_a = x0_steps[imin] / np.diff(amplitudes)[0]
        p0_b = x0[0] - p0_a * amplitudes[0]
        popt, _ = sciopt.curve_fit(curve, amplitudes, x0, p0=(p0_a, p0_b))
        rx_amp = -popt[1] / popt[0]

        analysis_results.append(AnalysisResultData(name='rx_amp', value=rx_amp))

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Rx amplitude',
                ylabel=r'$\theta_0$',
                ylim=(-np.pi - 0.1, np.pi + 0.1)
            )
            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)
            plotter.set_series_data(
                'angle_diff',
                x_formatted=amplitudes,
                y_formatted=x0,
                y_formatted_err=np.zeros_like(amplitudes),
                x_interp=x_interp,
                y_interp=curve(x_interp, *popt)
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class CycledRepeatedCRRxAmplitudeCal(BaseCalibrationExperiment, CycledRepeatedCRRxAmplitude):
    """Calibration experiment for CR width and Rx amplitude"""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        rcr_type: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['amp', 'sign_angle'],
        schedule_name: str = 'offset_rx',
        amplitudes: Optional[Sequence[float]] = None,
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
            rcr_type,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes
        )
        # Need to have circuits() return decomposed circuits because of how _transpiled_circuits
        # work in calibration experiments
        for qutrit_qpt in self.component_experiment():
            for qpt in qutrit_qpt.component_experiment():
                qpt.set_experiment_options(decompose_circuits=True)

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
