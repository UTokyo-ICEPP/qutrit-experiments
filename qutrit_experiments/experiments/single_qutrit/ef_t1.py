"""A T1 experiment for qutrits."""
import logging
from collections.abc import Sequence
from typing import Optional, Union
import warnings
import lmfit
from matplotlib.figure import Figure
import numpy as np
import scipy.linalg as scilin
from qiskit import QuantumCircuit
from qiskit.providers import Backend
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import (DATA_ENTRY_PREFIX,
                                                                   PARAMS_ENTRY_PREFIX)
from qiskit_experiments.framework import (AnalysisResultData, BackendTiming, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.framework.containers import ArtifactData

from ...experiment_mixins import MapToPhysicalQubits
from ...framework.ternary_mcm_analysis import TernaryMCMResultAnalysis
from ...gates import X12Gate

logger = logging.getLogger(__name__)


class EFT1(MapToPhysicalQubits, BaseExperiment):
    """A T1 experiment for qutrits."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delays = np.linspace(0., 5.e-4, 30)
        # Number of buffer circuits to insert to let the |2> state relax
        options.insert_buffer_circuits = 2
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.rep_delay = 5.e-4
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Optional[Union[list[float], np.array]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=EFT1Analysis(), backend=backend)
        if delays is not None:
            self.set_experiment_options(delays=delays)

    def circuits(self) -> list[QuantumCircuit]:
        timing = BackendTiming(self.backend)

        buffer_circuit = QuantumCircuit(1, 2)
        buffer_circuit.measure(0, 0)
        buffer_circuit.barrier(0)
        buffer_circuit.x(0)
        buffer_circuit.barrier(0)
        buffer_circuit.measure(0, 1)
        buffer_circuit.metadata = {'unit': None}

        circuits = []
        for delay in self.experiment_options.delays:
            circ = QuantumCircuit(1, 2)
            circ.x(0)
            circ.append(X12Gate(), [0])
            circ.barrier(0)
            circ.delay(timing.round_delay(time=delay), 0, timing.delay_unit)
            circ.barrier(0)
            circ.measure(0, 0)
            circ.barrier(0)
            circ.x(0)
            circ.append(X12Gate(), [0])
            circ.barrier(0)
            circ.measure(0, 1)
            circ.metadata = {'xval': timing.delay_time(time=delay), 'unit': 's'}
            circuits.append(circ)
            for idx in range(self.experiment_options.insert_buffer_circuits):
                circ = buffer_circuit.copy()
                circ.metadata['xval'] = idx
                circuits.append(circ)

        return circuits


def _three_component_relaxation(x, p0, p1, p2, g10, g20, g21):
    transition_matrix = np.array(
        [[0., g10, g20],
         [0., -g10, g21],
         [0., 0., -(g20 + g21)]]
    )
    return scilin.expm(np.asarray(x)[..., None, None] * transition_matrix) @ np.array([p0, p1, p2])


def _three_component_relaxation_jac(x, p0, p1, p2, g10, g20, g21):
    x = np.asarray(x)[..., None, None]
    transition_matrix = np.array(
        [[0., g10, g20],
         [0., -g10, g21],
         [0., 0., -(g20 + g21)]]
    )
    exp_mat = scilin.expm(x * transition_matrix)
    fnval = exp_mat @ np.array([p0, p1, p2])
    dmdg10 = x * np.array([[0., 1., 0.], [0., -1., 0.], [0., 0., 0.]])
    dmdg20 = x * np.array([[0., 0., 1.], [0., 0., 0.], [0., 0., -1.]])
    dmdg21 = x * np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., -1.]])
    return np.moveaxis(
        np.stack([
            exp_mat[..., :, 0],
            exp_mat[..., :, 1],
            exp_mat[..., :, 2],
            np.sum(dmdg10 * fnval[..., None, :], axis=-1),
            np.sum(dmdg20 * fnval[..., None, :], axis=-1),
            np.sum(dmdg21 * fnval[..., None, :], axis=-1)
        ], axis=-2),
        -1, -2
    )


def _prob_0(x, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(x, p0, p1, p2, g10, g20, g21)[..., 0]


def _prob_1(x, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(x, p0, p1, p2, g10, g20, g21)[..., 1]


def _prob_2(x, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(x, p0, p1, p2, g10, g20, g21)[..., 2]


def _variance(x, p0, p1, p2, g10, g20, g21, covar):
    jac = _three_component_relaxation_jac(x, p0, p1, p2, g10, g20, g21)
    return np.sum((jac @ covar)[..., :, None, :] * jac[..., None, :, :], axis=-1)


class EFT1Analysis(TernaryMCMResultAnalysis):
    """Run fit with rate equations."""

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.filter_data = {'unit': 's'}
        return options

    def __init__(self, name: Optional[str] = None):
        super().__init__(models=[
            lmfit.Model(_prob_0, name='0'),
            lmfit.Model(_prob_1, name='1'),
            lmfit.Model(_prob_2, name='2')
        ], name=name)

        self.set_options(
            result_parameters=[
                curve.ParameterRepr('g10', 'Γ10'),
                curve.ParameterRepr('g20', 'Γ20'),
                curve.ParameterRepr('g21', 'Γ21')
            ],
            bounds=({p: (-0.1, 1.1) for p in ['p0', 'p1', 'p2']}
                    | {p: (0., np.inf) for p in ['g10', 'g20', 'g21']}),
            p0={'p0': 0., 'p1': 0., 'p2': 1.}
        )

        self.options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Probability",
            xval_unit="s",
            ylim=(0., 1.)
        )

    def _run_analysis(
        self, experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Must override _run_analysis almost entirely because eval_with_uncertainty fails for
        complex non-expression models."""
        figures = []
        result_data = []
        artifacts = []

        # Flag for plotting can be "always", "never", or "selective"
        # the analysis option overrides self._generate_figures if set
        if self.options.get("plot", None):
            plot = "always"
        elif self.options.get("plot", None) is False:
            plot = "never"
        else:
            plot = getattr(self, "_generate_figures", "always")

        # Prepare for fitting
        self._initialize(experiment_data)

        table = self._format_data(self._run_data_processing(experiment_data.data()))
        formatted_subset = table.filter(category=self.options.fit_category)
        fit_data = self._run_curve_fit(formatted_subset)

        if fit_data.success:
            quality = self._evaluate_quality(fit_data)
        else:
            quality = "bad"

        # After the quality is determined, plot can become a boolean flag for whether
        # to generate the figure
        plot_bool = plot == "always" or (plot == "selective" and quality == "bad")

        if self.options.return_fit_parameters:
            # Store fit status overview entry regardless of success.
            # This is sometime useful when debugging the fitting code.
            overview = AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.name,
                value=fit_data,
                quality=quality,
                extra=self.options.extra,
            )
            result_data.append(overview)

        if fit_data.success:
            # Add fit data to curve data table
            model_names = self.model_names()
            for series_id, sub_data in formatted_subset.iter_by_series_id():
                xval = sub_data.x
                if len(xval) == 0:
                    # If data is empty, skip drawing this model.
                    # This is the case when fit model exist but no data to fit is provided.
                    continue
                # Compute X, Y values with fit parameters.
                xval_arr_fit = np.linspace(np.min(xval), np.max(xval), num=100, dtype=float)
                yval_arr_fit = self._models[series_id].eval(x=xval_arr_fit, **fit_data.params)
                yerr_arr_fit = np.zeros_like(xval_arr_fit)
                for xval, yval, yerr in zip(xval_arr_fit, yval_arr_fit, yerr_arr_fit):
                    table.add_row(
                        xval=xval,
                        yval=yval,
                        yerr=yerr,
                        series_name=model_names[series_id],
                        series_id=series_id,
                        category="fitted",
                        analysis=self.name,
                    )
            result_data.extend(
                self._create_analysis_results(
                    fit_data=fit_data,
                    quality=quality,
                    **self.options.extra.copy(),
                )
            )

        if self.options.return_data_points:
            # Add raw data points
            warnings.warn(
                f"{DATA_ENTRY_PREFIX + self.name} has been moved to experiment data artifacts. "
                "Saving this result with 'return_data_points'=True will be disabled in "
                "Qiskit Experiments 0.7.",
                DeprecationWarning,
            )
            result_data.extend(self._create_curve_data(curve_data=formatted_subset))

        artifacts.append(
            ArtifactData(
                name="curve_data",
                data=table,
            )
        )
        artifacts.append(
            ArtifactData(
                name="fit_summary",
                data=fit_data,
            )
        )

        if plot_bool:
            if fit_data.success:
                self.plotter.set_supplementary_data(
                    fit_red_chi=fit_data.reduced_chisq,
                    primary_results=[r for r in result_data if not r.name.startswith("@")],
                )
            figures.extend(self._create_figures(curve_data=table))

        buffer_data = [datum for datum in experiment_data.data()
                       if datum['metadata']['unit'] is None]
        if buffer_data:
            num_buffers = len(set(datum['metadata']['xval'] for datum in buffer_data))
            buffer_p2 = self.options.data_processor(buffer_data)[2::3].reshape(-1, num_buffers)
            result_data.append(
                AnalysisResultData(name='buffer_p2', value=buffer_p2)
            )

        return result_data + artifacts, figures

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable
    ) -> curve.FitOptions:
        data_1 = curve_data.filter(series='1')
        data_2 = curve_data.filter(series='2')
        p1_dominant = data_2.y < 0.2
        user_opt.p0.set_if_empty(
            g10=-curve.guess.exp_decay(data_1.x[p1_dominant], data_1.y[p1_dominant]),
            g20=0.,
            g21=-curve.guess.exp_decay(data_2.x, data_2.y)
        )
        return user_opt
