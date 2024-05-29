"""A T1 experiment for qutrits."""
import logging
from collections.abc import Sequence
from typing import Optional, Union
import lmfit
from matplotlib.figure import Figure
import numpy as np
import scipy.linalg as scilin
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.providers import Backend
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.framework import (AnalysisResultData, BackendTiming, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

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
        # Plotting fails for complex non-expression models (at eval_with_uncertainties)
        plot_option = self.options.plot
        self.options.plot = False

        results, figures = super()._run_analysis(experiment_data)

        buffer_data = [datum for datum in experiment_data.data()
                       if datum['metadata']['unit'] is None]
        if buffer_data:
            num_buffers = len(set(datum['metadata']['xval'] for datum in buffer_data))
            buffer_p2 = self.options.data_processor(buffer_data)[2::3].reshape(-1, num_buffers)
            results.append(
                AnalysisResultData(name='buffer_p2', value=buffer_p2)
            )

        self.options.plot = plot_option

        if plot_option:
            cls_name = self.__class__.__name__
            fit_params = next(res for res in results
                              if res.name == f'{PARAMS_ENTRY_PREFIX}{cls_name}').value
            for idx, model in enumerate(self._models):
                table = next(res for res in results if res.name == 'curve_data')
                sub_data = table.filter(category='formatted')
                self.plotter.set_series_data(
                    model._name,
                    x_formatted=sub_data.x,
                    y_formatted=sub_data.y,
                    y_formatted_err=sub_data.y_err
                )
                if not fit_params.success:
                    continue

                x_interp = np.linspace(np.min(sub_data['xdata']), np.max(sub_data['xdata']), 100)
                y_interp = model.eval(x=x_interp, **fit_params.params)
                variance = _variance(x_interp, covar=fit_params.covar, **fit_params.params)

                # Add fit line data
                self.plotter.set_series_data(
                    model._name,
                    x_interp=x_interp,
                    y_interp=y_interp,
                    y_interp_err=np.sqrt(variance[..., idx, idx])
                )

            figures.append(self.plotter.figure())

            if buffer_data:
                xdata = self.plotter.series_data['0']['x_formatted']
                ax = get_non_gui_ax()
                for ibuf in range(num_buffers):
                    ax.scatter(xdata, unp.nominal_values(buffer_p2[:, ibuf]), label=f'buffer{ibuf}')
                ax.set_xlabel('Preceeding delay (s)')
                ax.set_ylabel('P2')
                ax.legend()
                figures.append(ax.get_figure())

        return results, figures

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData
    ) -> curve.FitOptions:
        data_1 = curve_data.get_subset_of('1')
        data_2 = curve_data.get_subset_of('2')
        p1_dominant = data_2.y < 0.2
        user_opt.p0.set_if_empty(
            g10=-curve.guess.exp_decay(data_1.x[p1_dominant], data_1.y[p1_dominant]),
            g20=0.,
            g21=-curve.guess.exp_decay(data_2.x, data_2.y)
        )
        return user_opt
