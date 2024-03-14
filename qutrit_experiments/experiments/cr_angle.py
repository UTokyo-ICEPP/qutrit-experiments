"""CR angle calibration.

Sweep the CR angle with a small X tone.

U = exp(-i[(omega_CRx + omega_x) x + omega_CRy y] t)
"""
from collections.abc import Sequence
from typing import Any, Optional, Union
import lmfit
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import least_squares
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.curve_data import ParameterRepr
from qiskit_experiments.framework import AnalysisResultData, BaseExperiment, ExperimentData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..experiment_mixins import MapToPhysicalQubits
from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment


class CRAngle(MapToPhysicalQubits, BaseExperiment):
    """CR angle calibration.

    |<1| exp(-i/2 [(ωCRx + ωx) X + ωCRy Y] t) |0>|^2 = [sin(1/2 √[(ωCRx + ωx)^2 + ωCRy^2] t)]^2
    = 1/2 [1 - cos(√[ωCR^2 + ωx^2 + 2 ωCR cos(phi) ωx] t))]
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.control_state = None
        options.angle_param_name = 'angle'
        options.angles = np.linspace(-np.pi, np.pi, 32, endpoint=False)
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
        sched = self.experiment_options.schedule
        angle = sched.get_parameters(self.experiment_options.angle_param_name)[0]

        template = QuantumCircuit(2, 1)
        if self.experiment_options.control_state == 1:
            template.x(0)
        template.append(Gate('cr', 2, [angle]), [0, 1])
        template.measure(1, 0)
        template.add_calibration('cr', self.physical_qubits, sched, [angle])

        circuits = []
        for aval in self.experiment_options.angles:
            circuit = template.assign_parameters({angle: aval}, inplace=False)
            circuit.metadata = {'xval': aval}
            circuits.append(circuit)

        return circuits


class CRAngleAnalysis(curve.CurveAnalysis):
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="amp * cos(coeff * cos(x - x0) + phase) + base",
                    name="doublecos",
                )
            ],
            name=name
        )
        self.set_options(result_parameters=[ParameterRepr('x0', 'angle')])

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)
        min_abs_y, ix_min = curve.guess.min_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            amp=(-2 * max_abs_y, 2 * max_abs_y),
            phase=(-np.pi, np.pi),
            base=(-max_abs_y, max_abs_y),
        )
        p0_base = curve.guess.constant_sinusoidal_offset(curve_data.y)
        p0_amp = curve.guess.max_height(curve_data.y - p0_base, absolute=True)[0]
        arg_max = np.arccos((max_abs_y - p0_base) / p0_amp) # phase - coeff
        arg_min = np.arccos((min_abs_y - p0_base) / p0_amp) # phase + coeff
        p0_coeff = (arg_min - arg_max) / 2.
        p0_phase = (arg_min + arg_max) / 2.
        p0_x0 = curve_data.x[ix_min]

        user_opt.p0.set_if_empty(
            amp=p0_amp,
            base=p0_base,
            coeff=p0_coeff,
            phase=p0_phase,
            x0=p0_x0
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
