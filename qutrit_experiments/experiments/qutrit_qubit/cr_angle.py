"""CR angle calibration.

Sweep the CR angle with a small X tone.

U = exp(-i[(omega_CRx + omega_x) x + omega_CRy y] t)
"""
from collections.abc import Sequence
import logging
from typing import Any, Optional, Union
import lmfit
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import least_squares
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.curve_data import ParameterRepr
from qiskit_experiments.framework import AnalysisResultData, BaseExperiment, ExperimentData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...experiment_mixins import MapToPhysicalQubits
from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import X12Gate
from ...util.pulse_area import gs_effective_duration, rabi_cycles_per_area

logger = logging.getLogger(__name__)


class CRAngle(MapToPhysicalQubits, BaseExperiment):
    """CR angle calibration.

    |<1| exp(-i/2 [(ωCRx + ωx) X + ωCRy Y] t) |0>|^2 = [sin(1/2 √[(ωCRx + ωx)^2 + ωCRy^2] t)]^2
    = 1/2 [1 - cos(√[ωCR^2 + ωx^2 + 2 ωCR cos(φ) ωx] t))]
    = 1/2 [1 - cos(√[1 + sin(2θ)cos(φ)] Ωt))]  (ωx = Ω cos(θ), ωCR = Ω sin(θ))
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.control_state = None
        options.angle_param_name = 'angle'
        options.angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        control_state: int,
        angle_param_name: str = 'angle',
        angles: Optional[Sequence[float]] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, backend=backend)
        self.set_experiment_options(schedule=schedule, control_state=control_state,
                                    angle_param_name=angle_param_name)
        if angles is not None:
            self.set_experiment_options(angles=np.array(angles))
        self.analysis = CRAngleAnalysis()
        self.extra_metadata = extra_metadata

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata

    def circuits(self) -> list[QuantumCircuit]:
        angles = self.experiment_options.angles
        # Circuits for normalization
        c0 = QuantumCircuit(2)
        c0.metadata = {'series': 'spam-cal', 'xval': 0.}
        c1 = QuantumCircuit(2)
        c1.x(1)
        c1.metadata = {'series': 'spam-cal', 'xval': 1.}

        circuits = [c0, c1]

        sched = self.experiment_options.schedule
        angle = sched.get_parameters(self.experiment_options.angle_param_name)[0]

        template = QuantumCircuit(2, 1)
        if self.experiment_options.control_state != 0:
            template.x(0)
        if self.experiment_options.control_state == 2:
            template.append(X12Gate(), [0])
        template.append(Gate('cr', 2, [angle]), [0, 1])
        if self.experiment_options.control_state == 2:
            template.append(X12Gate(), [0])
        template.measure(1, 0)
        template.add_calibration('cr', self.physical_qubits, sched, [angle])

        for aval in angles:
            circuit = template.assign_parameters({angle: aval}, inplace=False)
            circuit.metadata = {'series': 'experiment', 'xval': aval}
            circuits.append(circuit)

        return circuits


class CRAngleAnalysis(curve.CurveAnalysis):
    """Analysis for CRAngle."""
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr='amp / 2. * (2. * x - 1.) + base',
                    name='spam-cal'
                ),
                lmfit.models.ExpressionModel(
                    expr='amp * cos(sqrt(1. + s * cos(x - x0)) * psi) + base',
                    name='experiment',
                )
            ],
            name=name
        )
        self.set_options(
            data_subfit_map={
                'spam-cal': {'series': 'spam-cal'},
                'experiment': {'series': 'experiment'}
            },
            result_parameters=[ParameterRepr('x0', 'angle')],
            bounds={
                'psi': (0., np.inf),
                's': (0., 1.),
                'x0': (-np.pi, np.pi)
            },
            fixed_parameters={
                # These values will be updated in _generate_fit_guesses if spam_cal data exist
                'amp': -0.5,
                'base': 0.5
            }
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        spam_cal_data = curve_data.get_subset_of('spam-cal')
        if spam_cal_data.x.shape[0] == 0:
            p0_base = 0.5
            p0_amp = -0.5
        else:
            p0_base = np.mean(spam_cal_data.y)
            p0_amp = p0_base - np.abs(np.diff(spam_cal_data.y))
            user_opt.p0['base'] = p0_base
            user_opt.p0['amp'] = p0_amp

        exp_data = curve_data.get_subset_of('experiment')
        cos_val = (exp_data.y - p0_base) / p0_amp
        cos_arg = np.arccos(np.maximum(np.minimum(cos_val, 1.), -1.))
        max_cos_arg_sq = np.square(np.amax(cos_arg)) # (1 + s) * psi^2
        min_cos_arg_sq = np.square(np.amin(cos_arg)) # (1 - s) * psi^2
        psi2 = (max_cos_arg_sq + min_cos_arg_sq) / 2.
        p0_psi = np.sqrt(psi2)
        p0_s = max_cos_arg_sq / psi2 - 1.
        p0_x0 = exp_data.x[np.argmax(cos_arg)]

        user_opt.p0.set_if_empty(
            x0=p0_x0,
            psi=p0_psi,
            s=p0_s
        )

        return user_opt


class CRAngleCounterScan(BatchExperiment):
    """CRAngle scanning the counter angle."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.counter_angles = np.linspace(-np.pi / 8., np.pi / 8., 5)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        control_state: int,
        angle_param_name: str = 'angle',
        counter_angle_param_name: str = 'counter_angle',
        angles: Optional[Sequence[float]] = None,
        counter_angles: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if (counter_angles_exp := counter_angles) is None:
            counter_angles_exp = self._default_experiment_options().counter_angles

        counter_angle_param = schedule.get_parameters(counter_angle_param_name)[0]
        experiments = []
        for counter_angle in counter_angles_exp:
            sched = schedule.assign_parameters({counter_angle_param: counter_angle}, inplace=False)
            experiments.append(
                CRAngle(physical_qubits, sched, control_state,
                        angle_param_name=angle_param_name, angles=angles, backend=backend,
                        extra_metadata={'counter_angle': counter_angle})
            )

        super().__init__(experiments, backend=backend,
                         analysis=CRAngleCounterScanAnalysis([exp.analysis for exp in experiments]))


class CRAngleCounterScanAnalysis(CompoundAnalysis):
    """Analysis for CRAngleCounterScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = None
        options.plot = True
        return options

    def __init__(self, analyses: list[CRAngleAnalysis]):
        super().__init__(analyses, flatten_results=False)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        xvals = []
        yvals = []
        for idx in experiment_data.metadata['component_child_index']:
            child_data = experiment_data.child_data(idx)
            xvals.append(child_data.metadata['counter_angle'])
            yvals.append(child_data.analysis_results('angle').value)

        xvals = np.array(xvals)
        yvals = np.array(yvals)
        yvals_n = unp.nominal_values(yvals)
        yvals_e = unp.std_devs(yvals)

        def model(params, x):
            return params[0] * x + params[1]

        def residual(params, x, y, yerr):
            return (model(params, x) - y) / yerr

        def jacobian(params, x, y, yerr):
            return np.stack([x, np.ones_like(x)], axis=1) / yerr[:, None]

        p0 = (1., np.mean(yvals_n - xvals))
        result = least_squares(residual, p0, jac=jacobian, args=(xvals, yvals_n, yvals_e))
        popt = result.x
        approx_pcov = np.linalg.inv(result.jac.T @ result.jac) * 2.
        popt_ufloats = correlated_values(popt, approx_pcov)

        analysis_results.append(AnalysisResultData(name='angle', value=popt_ufloats[1]))

        if self.options.plot:
            x_interp = np.linspace(xvals[0], xvals[-1], 100)
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Counter angle',
                ylabel='CR angle'
            )
            plotter.set_series_data(
                'angles',
                x_formatted=xvals,
                y_formatted=yvals_n,
                y_formatted_err=yvals_e,
                x_interp=x_interp,
                y_interp=model(popt, x_interp)
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class FineCRAngleCal(BaseCalibrationExperiment, CRAngleCounterScan):
    """Calibration of CR angle using CRAngleCounterScan."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        control_state: int,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'cr_base_angle',
        schedule_name: str = 'cr',
        auto_update: bool = True,
        angles: Optional[Sequence[float]] = None,
        counter_angles: Optional[Sequence[float]] = None
    ):
        width = 256
        duration = gs_effective_duration(calibrations, physical_qubits, 'cr', width=width)
        # CR at the current width is expected to generate at most pi/2 - let counter give 2pi/5
        counter_amp = 0.2 / (rabi_cycles_per_area(backend, physical_qubits[1]) * duration)

        assign_params = {
            cal_parameter_name: Parameter('angle'),
            'cr_stark_amp': 0.,
            'counter_amp': counter_amp,
            'counter_base_angle': Parameter('counter_angle'),
            'counter_stark_amp': 0.,
            'width': width
        }
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            control_state,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            angle_param_name='angle',
            counter_angle_param_name='counter_angle',
            angles=angles,
            counter_angles=counter_angles
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        BaseUpdater.update(
            self._cals, experiment_data, self._param_name, schedule=self._sched_name,
            group=self.experiment_options.group, fit_parameter='angle'
        )
