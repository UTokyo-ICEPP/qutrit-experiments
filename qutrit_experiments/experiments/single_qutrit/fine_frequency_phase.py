"""Fine frequency calibration through interpolation of RamseyPhaseSweep results."""

from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
import scipy.optimize as sciopt
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from .delay_phase_offset import EFRamseyPhaseSweep, RamseyPhaseSweepAnalysis
from ...framework.calibration_updaters import EFFrequencyUpdater
from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import SetF12Gate
from ...util.matplotlib import make_list_plot

twopi = 2. * np.pi


class EFRamseyFrequencyScan(BatchExperiment):
    """Fine frequency calibration through interpolation of RamseyPhaseSweep results."""
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
            experiment.set_experiment_options(pre_schedule=(SetF12Gate(frequency), [0]))
            experiments.append(experiment)
            experiment.analysis.set_options(nwind_hypotheses=[0])
            analyses.append(experiment.analysis)

        analysis = EFRamseyFrequencyScanAnalysis(analyses)

        super().__init__(experiments, backend=backend, analysis=analysis)

        if delay_duration is not None:
            self.set_experiment_options(delay_duration=delay_duration)


class EFRamseyFrequencyScanAnalysis(CompoundAnalysis):
    """Interpolation analysis."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.data_processor = None
        options.common_amp = True
        options.nwind_hypotheses = [0]
        return options

    @classmethod
    def _propagated_option_keys(cls) -> list[str]:
        keys = super()._propagated_option_keys()
        return keys + ['common_amp', 'nwind_hypotheses']

    def __init__(
        self,
        analyses: list[RamseyPhaseSweepAnalysis]
    ):
        super().__init__(analyses, flatten_results=False)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata['component_child_index']
        omega_zs = []
        frequencies = []

        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            omega_zs.append(child_data.analysis_results('omega_z_dt').value
                            / child_data.metadata['dt'])
            frequencies.append(child_data.metadata['f12'])

        omega_zs = np.array(omega_zs)
        yval = unp.nominal_values(omega_zs / twopi)
        frequencies = np.array(frequencies)

        def func(f, f0, slope):
            return slope * (f - f0)

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

            figures.append(
                make_list_plot(experiment_data,
                               title_fn=lambda idx: f'f12 = {frequencies[idx] * 1.e-6:.1f} MHz')
            )

        return analysis_results, figures


class EFRamseyFrequencyScanCal(BaseCalibrationExperiment, EFRamseyFrequencyScan):
    """Calibration experiment for EFRamseyFrequencyScan."""
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

        f12_est = calibrations.get_parameter_value('f12', physical_qubits[0])
        frequencies = detunings + f12_est

        super().__init__(
            calibrations,
            physical_qubits,
            frequencies,
            schedule_name=None,
            delay_duration=delay_duration,
            num_points=num_points,
            backend=backend,
            cal_parameter_name='f12',
            auto_update=auto_update
        )

        self._updater = EFFrequencyUpdater

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass
