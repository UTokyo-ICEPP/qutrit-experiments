"""Initial CR amplitude determination.

Nonlinear effect in CR has been observed to break the block-diagonality. We check the decay of
control=|2> as a function of the amplitude and determine an acceptable amplitude.
"""
from collections.abc import Sequence
import logging
from typing import Optional, Union
from matplotlib.figure import Figure
import lmfit
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.curve_data import ParameterRepr
from qiskit_experiments.data_processing import DataProcessor, MarginalizeCounts, Probability
from qiskit_experiments.framework import AnalysisResultData, BaseExperiment, ExperimentData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...data_processing import get_ternary_data_processor
from ...experiment_mixins import MapToPhysicalQubits
from ...framework.ternary_mcm_analysis import TernaryMCMResultAnalysis
from ...gates import CrossResonanceGate, X12Gate

logger = logging.getLogger(__name__)


class CRDiagonality(MapToPhysicalQubits, BaseExperiment):
    """CR characterization.

    Experiment: Raise the control qubit to |2>, run the CR tone, and measure the control state.
    Mid-circuit measurement is used to discriminate |2> from |0> and |1>:
    state result
    2     11
    1     01
    0     10
    Measure the Rabi oscillation on the target at the same time to determine the saturation point
    of X.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.amplitudes = np.linspace(0., 0.9, 10)
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.rep_delay = 5.e-4
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: Optional[ScheduleBlock] = None,
        amplitudes: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=CRDiagonalityAnalysis(), backend=backend)
        self.set_experiment_options(schedule=schedule)
        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)

    def circuits(self) -> list[QuantumCircuit]:
        if (sched := self.experiment_options.schedule) is None:
            amplitude = Parameter('amplitude')
        else:
            amplitude = next(iter(sched.parameters))

        template = QuantumCircuit(2, 3)
        template.x(0)
        template.append(X12Gate(), [0])
        template.append(CrossResonanceGate(params=[amplitude]), [0, 1])
        template.measure(1, 0)
        template.measure(0, 1)
        template.x(0)
        template.append(X12Gate(), [0])
        template.measure(0, 2)
        if sched is not None:
            template.add_calibration(CrossResonanceGate.gate_name, self.physical_qubits, sched,
                                     params=[amplitude])

        circuits = []
        for aval in self.experiment_options.amplitudes:
            circ = template.assign_parameters({amplitude: aval}, inplace=False)
            circ.metadata = {'xval': aval}
            circuits.append(circ)

        return circuits


class CRDiagonalityAnalysis(TernaryMCMResultAnalysis):
    """Analysis for CRDiagonality."""
    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr='c0 + 0 * x',
                    name='model0'
                ),
                lmfit.models.ExpressionModel(
                    expr='a * x ** 2 + b1',
                    name='model1'
                ),
                lmfit.models.ExpressionModel(
                    expr='-a * x ** 2 + b2',
                    name='model2',
                )
            ],
            name=name
        )
        self.set_options(
            result_parameters=[ParameterRepr('a', 'curvature')],
            bounds={'a': (0., np.inf)}
        )
        self.plotter.set_figure_options(
            xlabel='CR amplitude',
            ylabel='Qutrit state probability',
            ylim=(-0.05, 1.05)
        )

    def _initialize(self, experiment_data: ExperimentData):
        if (data_processor := self.options.data_processor) is None:
            data_processor = get_ternary_data_processor(
                assignment_matrix=self.options.assignment_matrix,
                include_invalid=False,
                serialize=True
            )
        if not isinstance(data_processor._nodes[0], MarginalizeCounts):
            data_processor._nodes.insert(0, MarginalizeCounts({1, 2}))
        self.options.data_processor = data_processor

        super()._initialize(experiment_data)

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        data0 = curve_data.get_subset_of('model0')
        data1 = curve_data.get_subset_of('model1')
        data2 = curve_data.get_subset_of('model2')
        bmax = np.max(data1.y + data2.y)
        user_opt.bounds.set_if_empty(b1=(0., bmax), b2=(0., bmax))
        p0_b1 = data1.y[0]
        p0_b2 = data2.y[0]
        p0_a = (data1.y[-1] - p0_b1) / data1.x[-1] ** 2
        p0_c0 = np.mean(data0.y)
        user_opt.p0.set_if_empty(a=p0_a, b1=p0_b1, b2=p0_b2, c0=p0_c0)

        return user_opt

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        results, figures = super()._run_analysis(experiment_data)

        # Now make a plot for the target qubit
        if self.options.plot:
            data_processor = DataProcessor('counts', [MarginalizeCounts({0}), Probability('1')])
            tdata = data_processor(experiment_data.data())
            xdata = np.asarray([datum["metadata"]['xval'] for datum in experiment_data.data()],
                            dtype=float)

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='CR amplitude',
                ylabel='Target P(1)'
            )
            plotter.set_series_data(
                'target',
                x_formatted=xdata,
                y_formatted=unp.nominal_values(tdata),
                y_formatted_err=unp.std_devs(tdata)
            )
            figures.append(plotter.figure())

        return results, figures


class CRInitialAmplitudeCal(BaseCalibrationExperiment, CRDiagonality):
    """Calibration of CR amplitude using CRDiagonality."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.max_amp = 0.8
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'cr_amp',
        schedule_name: str = 'cr',
        auto_update: bool = True,
        cutoff: float = 0.99,
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[int] = None
    ):
        if width is None:
            width = 256

        self._amplitude = Parameter('amplitude')
        assign_params = {
            cal_parameter_name: self._amplitude,
            'cr_stark_amp': 0.,
            'counter_amp': 0.,
            'counter_stark_amp': 0.,
            'width': width
        }
        self._schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                                   assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amplitudes=amplitudes
        )
        self._cutoff = cutoff

    def _attach_calibrations(self, circuit: QuantumCircuit):
        amplitude = circuit.metadata['xval']
        sched = self._schedule.assign_parameters({self._amplitude: amplitude}, inplace=False)
        circuit.add_calibration(CrossResonanceGate.gate_name, self.physical_qubits, sched,
                                params=[amplitude])

    def update_calibrations(self, experiment_data: ExperimentData):
        curvature = BaseUpdater.get_value(experiment_data, 'curvature')
        cr_amp = min(self.experiment_options.max_amp, np.sqrt((1. - self._cutoff) / curvature))
        BaseUpdater.add_parameter_value(self._cals, experiment_data, cr_amp, self._param_name,
                                        schedule=self._sched_name,
                                        group=self.experiment_options.group)
