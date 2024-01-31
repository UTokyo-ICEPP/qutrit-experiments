from collections.abc import Sequence
from typing import Optional
import numpy as np
import scipy.optimize as sciopt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..qutrit_qubit_tomography import QutritQubitTomographyScan, QutritQubitTomographyScanAnalysis
from .util import RCRType, get_margin, make_crcr_circuit

twopi = 2. * np.pi


class CycledRepeatedCRWidth(QutritQubitTomographyScan):
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
        rcr_type: RCRType,
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


class CycledRepeatedCRWidthAnalysis(QutritQubitTomographyScanAnalysis):
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
            return (np.asarray(w) * a + b) % twopi

        widths = next(res.value for res in analysis_results if res.name == 'crp_width')
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        xdiffs = (unitaries[:, 1, 0] - unitaries[:, 0, 0]) % twopi
        xdiff_steps = np.diff(xdiffs)
        imin = np.argmin(np.abs(xdiff_steps))
        p0_a = xdiff_steps[imin] / np.diff(widths)[0]
        p0_b = xdiffs[0] - p0_a * widths[0]
        popt, _ = sciopt.curve_fit(curve, widths, xdiffs, p0=(p0_a, p0_b))
        slope, intercept = popt
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
            y_interp = (curve(x_interp, slope, intercept) + np.pi) % twopi - np.pi
            plotter.set_series_data(
                'angle_diff',
                x_formatted=widths,
                y_formatted=(unitaries[:, 1, 0] - unitaries[:, 0, 0] + twopi) % (2. * twopi) - twopi,
                y_formatted_err=np.zeros_like(widths),
                x_interp=x_interp,
                y_interp=y_interp
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class CycledRepeatedCRWidthCal(BaseCalibrationExperiment, CycledRepeatedCRWidth):
    """Calibration experiment for CR width and Rx amplitude"""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
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
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            width_param_name=cal_parameter_name[0],
            margin_param_name=cal_parameter_name[1],
            widths=widths
        )

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
