from typing import Any, Dict, Optional, Sequence, Union, List, Tuple
import numpy as np
from uncertainties import correlated_values, unumpy as unp
import scipy.optimize as sciopt
from matplotlib.figure import Figure
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options, ExperimentData, AnalysisResultData, BackendData
from qiskit_experiments.framework.matplotlib import default_figure_canvas
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.library import FineFrequency
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..common.framework_overrides import CompoundAnalysis, BatchExperiment
from ..common.linked_curve_analysis import LinkedCurveAnalysis
from ..common.iq_classification import IQClassification
from ..common.util import default_shots
from .delay_phase_offset import EFRamseyPhaseSweep, RamseyPhaseSweepAnalysis
from .dummy_data import ef_memory, single_qubit_counts

twopi = 2. * np.pi


class EFRamseyFrequencyScan(IQClassification, BatchExperiment):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delay_duration = 160
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Sequence[float],
        delay_duration: Optional[int] = None,
        num_points: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        if (delay_duration_exp := delay_duration) is None:
            delay_duration_exp = self._default_experiment_options().delay_duration

        experiments = []
        analyses = []

        for frequency in frequencies:
            experiment = EFRamseyPhaseSweep(physical_qubits,
                                            delay_durations=[0, delay_duration_exp],
                                            num_points=num_points,
                                            extra_metadata={'f12': frequency},
                                            backend=backend)
            experiment.set_experiment_options(f12=frequency)
            experiments.append(experiment)
            experiment.analysis.set_options(nwind_hypotheses=[0])
            analyses.append(experiment.analysis)

        analysis = EFRamseyFrequencyScanAnalysis(analyses)

        super().__init__(experiments, backend=backend, analysis=analysis)

        if delay_duration is not None:
            self.set_experiment_options(delay_duration=delay_duration)

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[np.ndarray]:
        mean_f12 = np.mean([exp.experiment_options.f12 for exp in self._experiments])

        data = []
        for exp in self._experiments:
            orig = exp.experiment_options.dummy_omega_z
            exp.experiment_options.dummy_omega_z = (exp.experiment_options.f12 - mean_f12) * twopi
            data += exp.dummy_data(None)
            exp.experiment_options.dummy_omega_z = orig

        return data


class EFRamseyFrequencyScanAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def __init__(
        self,
        analyses: List[RamseyPhaseSweepAnalysis]
    ):
        super().__init__(analyses, flatten_results=False)

        self.list_figure = Figure(figsize=[9.6, 1.2 * len(analyses)])
        _ = default_figure_canvas(self.list_figure)
        axs = self.list_figure.subplots(len(analyses), 1, sharex=True)
        for analysis, ax in zip(analyses, axs):
            analysis.plotter.set_options(axis=ax)

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        for subanalysis in self._analyses:
            subanalysis.options.plot = self.options.plot

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        component_index = experiment_data.metadata['component_child_index']
        omega_zs = []
        frequencies = []

        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            omega_zs.append(child_data.analysis_results('omega_z_dt').value / child_data.metadata['dt'])
            frequencies.append(child_data.metadata['f12'])

        omega_zs = np.array(omega_zs)
        yval = unp.nominal_values(omega_zs / twopi)
        frequencies = np.array(frequencies)

        func = lambda f, f0, slope: slope * (f - f0)

        p0 = (
            np.mean(frequencies),
            (yval[-1] - yval[0]) / (frequencies[-1] - frequencies[0])
        )

        popt, pcov = sciopt.curve_fit(func, frequencies, yval, p0=p0)
        popt_ufloats = correlated_values(nom_values=popt, covariance_mat=pcov, tags=['f0', 'slope'])

        analysis_results.append(AnalysisResultData(name='f12', value=popt_ufloats[0]))

        if self.options.plot:
            ## Plot phase offset differences
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel=f'Frequency - {popt[0] * 1.e-6:.2f} MHz',
                ylabel=r'$f_{drive} - f_{qubit}$',
                xval_unit='Hz'
            )
            plotter.set_series_data(
                'freq_scan',
                x_formatted=(frequencies - popt[0]),
                y_formatted=yval,
                y_formatted_err=unp.std_devs(omega_zs / twopi)
            )
            interp_x = np.linspace(frequencies[0], frequencies[-1], 100)
            plotter.set_series_data(
                'freq_scan',
                x_interp=(interp_x - popt[0]),
                y_interp=func(interp_x, *popt)
            )

            figures.append(plotter.figure())

            ## Set the titles of the list plot components
            for ax, frequency in zip(self.list_figure.axes, frequencies):
                ax.set_title(f'f12 = {frequency * 1.e-6:.1f} MHz')

            self.list_figure.tight_layout()

            figures.append(self.list_figure)

        return analysis_results, figures


class EFFrequencyUpdater(BaseUpdater):
    __fit_parameter__ = 'f12'


class EFRamseyFrequencyScanCal(BaseCalibrationExperiment, EFRamseyFrequencyScan):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        detunings: Optional[Sequence[float]] = None,
        delay_duration: Optional[int] = None,
        num_points: Optional[int] = None,
        auto_update: bool = True
    ):
        if detunings is None:
            detunings = np.linspace(-5.e+5, 5.e+5, 6)

        f12_est = calibrations.get_parameter_value('f12', physical_qubits[0], 'set_f12')
        frequencies = detunings + f12_est

        super().__init__(
            calibrations,
            physical_qubits,
            frequencies,
            schedule_name='set_f12',
            delay_duration=delay_duration,
            num_points=num_points,
            backend=backend,
            cal_parameter_name='f12',
            auto_update=auto_update
        )

        self._updater = EFFrequencyUpdater

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass


class TestFineFrequency(FineFrequency):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.pre_schedule = None
        return options

    def _pre_circuit(self):
        circ = QuantumCircuit(1)

        pre_schedule = self.experiment_options.pre_schedule
        if pre_schedule is not None:
            circ.append(Gate('pre_schedule', 1, []), [0])
            circ.add_calibration('pre_schedule', self.physical_qubits, pre_schedule)

        return circ
