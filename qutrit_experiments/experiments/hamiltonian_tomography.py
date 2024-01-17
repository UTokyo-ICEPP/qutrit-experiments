"""Hamiltonian tomography experiment."""
from collections.abc import Callable, Iterable, Sequence
from itertools import product
import logging
from typing import Any, Optional, Union
import lmfit
import numpy as np
import scipy.optimize as sciopt
from uncertainties import ufloat, correlated_values, unumpy as unp

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.result import Counts
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.framework import Options, ExperimentData, AnalysisResultData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..analyses.linked_curve_analysis import LinkedCurveAnalysis
from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from ..util.bloch import (amp_matrix, amp_func, phase_matrix, phase_func, base_matrix, base_func,
                          unit_bound, pos_unit_bound)
from ..util.polynomial import sparse_poly_fitfunc, PolynomialOrder
from .gs_rabi import GSRabi, GSRabiAnalysis

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


class HamiltonianTomography(BatchExperiment):
    r"""Identify the Bloch-space rotation matrix of the qubit under a GaussianSquare tone.

    Denote the effective Hamiltonian of a constant drive on a qubit as

    .. math::

        H = \frac{\omega^x}{2} x + \frac{\omega^y}{2} y + \frac{\omega^z}{2} z.

    The evolution of the expectation value of :math:`g` (:math:`g=x,y,z`) is given by

    .. math::

        \frac{d}{dt} \langle g \rangle (t) = i\langle [H, g] \rangle.

    For :math:`r(t) := (\langle x \rangle (t), \langle y \rangle (t), \langle z \rangle (t))^T`, we
    have

    .. math::

        r(t) = e^{G t} r(0)

    where

    .. math::

        G = \begin{pmatrix}
            0 & -\omega^z & \omega^y \\
            \omega^z & 0 & -\omega^x \\
            -\omega^y & \omega^x & 0
            \end{pmatrix}.

    Exponentiating this, we have

    .. math::

        e^{G t} = \frac{1}{\Omega^2} \begin{pmatrix}
        (\omega^x)^2 + [(\omega^y)^2 + (\omega^z)^2] \cos(\Omega t) &
        \omega^x \omega^y [1 - \cos(\Omega t)] - \Omega \omega^z \sin(\Omega t) &
        \omega^x \omega^z [1 - \cos(\Omega t)] + \Omega \omega^y \sin(\Omega t) \\
        \omega^y \omega^x [1 - \cos(\Omega t)] + \Omega \omega^z \sin(\Omega t) &
        (\omega^y)^2 + [(\omega^z)^2 + (\omega^x)^2] \cos(\Omega t) &
        \omega^y \omega^z [1 - \cos(\Omega t)] - \Omega \omega^x \sin(\Omega t) \\
        \omega^z \omega^x [1 - \cos(\Omega t)] - \Omega \omega^y \sin(\Omega t) &
        \omega^z \omega^y [1 - \cos(\Omega t)] + \Omega \omega^x \sin(\Omega t) &
        (\omega^z)^2 + [(\omega^x)^2 + (\omega^y)^2] \cos(\Omega t)
        \end{pmatrix},

    where :math:`\Omega := \sqrt{\sum_g (\omega^g)^2}.` Each element of :math:`e^{G t}` thus evolves
    sinusoidally with the same frequency. Reparametrizing
    :math:`\omega^g` as

    .. math::

        \omega^x/\Omega & = \sin \psi \cos \phi, \\
        \omega^y/\Omega & = \sin \psi \sin \phi, \\
        \omega^z/\Omega & = \cos \psi,

    we arrive at

    .. math::

        e^{G t} = \begin{pmatrix}
        (1 - \sin^2 \psi \cos^2 \phi) \cos(\Omega t) + \sin^2 \psi \cos^2 \phi &
        -\sin^2 \psi \sin \phi \cos \phi \cos(\Omega t) - \cos \psi \sin(\Omega t)
            + \sin^2 \psi \sin \phi \cos \phi &
        \sin \psi [-\cos \psi \cos \phi \cos(\Omega t) + \sin \phi \sin(\Omega t)]
            + \sin \psi \cos \psi \cos \phi \\
        -\sin^2 \psi \sin \phi \cos \phi \cos(\Omega t) + \cos \psi \sin(\Omega t)
            + \sin^2 \psi \sin \phi \cos \phi &
        (1 - \sin^2 \psi \sin^2 \phi) \cos(\Omega t) + \sin^2 \psi \sin^2 \phi &
        \sin \psi [-\cos \psi \sin \phi \cos(\Omega t) - \cos \phi \sin(\Omega t)]
            + \sin \psi \cos \psi \sin \phi \\
        \sin \psi [-\cos \psi \cos \phi \cos(\Omega t) - \sin \phi \sin(\Omega t)]
            + \sin \psi \cos \psi \cos \phi &
        \sin \psi [-\cos \psi \sin \phi \cos(\Omega t) + \cos \phi \sin(\Omega t)]
            + \sin \psi \cos \psi \sin \phi &
        \sin^2 \psi \cos(\Omega t) + \cos^2 \psi
        \end{pmatrix}.

    Casting all elements of the matrix into the standard OscillationAnalysis form
    :math:`a \cos (2 \pi f t + \Phi) + b`,

    .. math::

        r(t) = \begin{pmatrix}
        (1 - \sin^2 \psi \cos^2 \phi) \cos(2 \pi f t) + \sin^2 \psi \cos^2 \phi &
        C \cos(2 \pi f t - \gamma) + \sin^2 \psi \sin \phi \cos \phi &
        \sin \psi A \cos(2 \pi f t + \alpha) + \sin \psi \cos \psi \cos \phi \\
        C \cos(2 \pi f t + \gamma) + \sin^2 \psi \sin \phi \cos \phi &
        (1 - \sin^2 \psi \sin^2 \phi) \cos(2 \pi f t) + \sin^2 \psi \sin^2 \phi &
        \sin \psi B \cos(2 \pi f t + \beta) + \sin \psi \cos \psi \sin \phi \\
        A \cos (2 \pi f t - \alpha) + \sin \psi \cos \psi \cos \phi &
        B \cos(2 \pi f t - \beta) + \sin \psi \cos \psi \sin \phi &
        \sin^2 \psi \cos(2 \pi f t) + \cos^2 \psi
        \end{pmatrix},

    where

    .. math::

        f & = \Omega / (2 \pi) \\
        A & = \sqrt{\cos^2 \psi \cos^2 \phi + \sin^2 \phi}, \\
        B & = \sqrt{\cos^2 \psi \sin^2 \phi + \cos^2 \phi}, \\
        C & = \sqrt{\sin^4 \psi \sin^2 \phi \cos^2 \phi + \cos^2 \psi}

    and

    .. math::

        \cos \alpha & = -\cos \psi \cos \phi / A, \\
        \sin \alpha & = -\sin \phi / A, \\
        \cos \beta & = -\cos \psi \cos \phi / B, \\
        \sin \beta & = \cos \phi / B \\
        \cos \gamma & = -\sin^2 \psi \sin \phi \cos \phi / C \\
        \sin \gamma & = -\cos \psi / C.

    Because a flat-top drive still must have pulse rise and fall times, the actual expectation
    value vector will be of form

    .. math::

        r'(t) := U_f e^{G t} U_i r'(0)

    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.dummy_components = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        rabi_init: Optional[Callable] = None,
        measured_logical_qubit: Optional[int] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        if rabi_init is None:
            rabi_init = GSRabi

        for idx, (init, meas_basis) in enumerate(product(['z', 'x'], ['x', 'y', 'z'])):
            exp = rabi_init(physical_qubits, schedule, widths=widths, initial_state=init,
                            meas_basis=meas_basis, time_unit=time_unit, experiment_index=idx,
                            backend=backend)
            exp.extra_metadata['basis'] = idx
            if measured_logical_qubit is not None:
                exp.set_experiment_options(measured_logical_qubit=measured_logical_qubit)

            # Bound the frequency to be unsigned. To allow the fit value of 0 comfortably, the lower
            # bound is set slightly negative.
            exp.analysis.options.bounds['freq'] = (-1.e+3, np.inf)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=HamiltonianTomographyAnalysis(analyses))
        self.extra_metadata = {}

    @property
    def num_circuits(self) -> int:
        return sum(subexp.num_circuits for subexp in self.component_experiment())

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        hamiltonian_components = self.experiment_options.dummy_components
        if hamiltonian_components is None:
            hamiltonian_components = np.array([-1857082.,  -1232100., -138360.])

        counts_list = []
        icirc = 0

        for imeas in range(3):
            subexp = self.component_experiment(imeas)
            options = subexp.experiment_options
            dummy_components = options.dummy_components
            options.dummy_components = hamiltonian_components
            counts_list.extend(
                subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits])
            )
            options.dummy_components = dummy_components
            icirc += subexp.num_circuits

        return counts_list


class HamiltonianTomographyAnalysis(CompoundAnalysis, curve.CurveAnalysis):
    """Analysis for HamiltonianTomography.

    
    """
    @staticmethod
    def evolution_factory(init, meas):
        def evolution(x, freq, psi, phi, theta, chi, kappa):
            xdims = tuple(range(len(np.asarray(x).shape)))
            x = np.expand_dims(x, (-2, -1))

            amp = np.expand_dims(amp_matrix(chi, kappa), xdims)
            phase = np.expand_dims(phase_matrix(chi, kappa, 0.), xdims)
            base = np.expand_dims(base_matrix(chi, kappa), xdims)
            matrix = amp * np.cos(theta + phase) + base

            amp = np.expand_dims(amp_matrix(psi, phi), xdims)
            phase = np.expand_dims(phase_matrix(psi, phi, 0.), xdims)
            base = np.expand_dims(base_matrix(psi, phi), xdims)
            matrix = (amp * np.cos(twopi * freq * x + phase) + base) @ matrix

            amp = np.expand_dims(amp_matrix(chi, kappa), xdims)
            phase = np.expand_dims(phase_matrix(chi, kappa, 0.), xdims)
            base = np.expand_dims(base_matrix(chi, kappa), xdims)
            matrix = (amp * np.cos(-theta + phase) + base) @ matrix

            # Need moveaxis + [meas, init] instead of [..., meas, init] because the latter
            # returns an ndarray object even when x is a scalar
            return np.moveaxis(matrix, (-2, -1), (0, 1))[meas, init]

        return evolution

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.primary_init = 2
        return options

    def __init__(self, analyses: list[GSRabiAnalysis]):
        super().__init__(analyses, flatten_results=False)

        self.set_options(
            bounds={
                'psi': (-1.e-3, np.pi + 1.e-3),
                'phi': (-twopi, twopi),
                'freq': (-1.e+5, np.inf), # giving some slack on the negative side
                'theta': (-1.e-3, twopi + 1.e-3),
                'chi': (-1.e-3, np.pi + 1.e-3),
                'kappa': (-twopi, twopi)
            }
        )
        self.plotter.set_figure_options(
            ylim=(-1.1, 1.1)
        )

        # Fit results of individual components
        self._component_results = None

    def _run_analysis(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata["component_child_index"]

        axes = ['x', 'y', 'z']

        # Identify the unitary of w=0
        # Solve for rise and fall: U = (cosθ/2 - isinθ/2(n.σ)) (cosθ/2 - isinθ/2(m.σ))
        # where n is a mirror image of m about the plane that contains the poles and the Hamiltonian
        # axes (ω(t)): (ωx(t), ωy(t)) is assumed to stay in the same plane during the rise-fall
        # evolution

        self._models = []
        subfit_map = {}
        for idx in range(len(self._analyses)):
            child_data = experiment_data.child_data(component_index[idx])
            basis = child_data.metadata['basis']
            init = basis // 3
            meas = basis % 3
            model = lmfit.Model(self.evolution_factory(init, meas),
                                name=f'{axes[meas]}|{axes[init]}')
            self._models.append(model)
            subfit_map[model._name] = {'experiment_index': idx}

        self.set_options(
            data_subfit_map=subfit_map,
            outcome='0',
            data_processor=self._analyses[0].options.data_processor
        )

        return super()._run_analysis(experiment_data)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        return curve.CurveAnalysis._run_analysis(self, experiment_data)

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        # CurveAnalysis._initialize
        super()._initialize(experiment_data)

        # "Flatten" the metadata by extracting the composite_metadata at index self._experiment_idx
        # CurveAnalysis machinery assumes x values and data filter keys to be in
        # the top-level metadata
        for datum in experiment_data.data():
            parent_metadata = datum['metadata']
            child_metadata = parent_metadata['composite_metadata'][0]
            for key, value in child_metadata.items():
                if key not in parent_metadata:
                    parent_metadata[key] = value

        results = {}
        bases = []
        component_index = experiment_data.metadata['component_child_index']
        for analysis, child_index in zip(self._analyses, component_index):
            result_name = PARAMS_ENTRY_PREFIX + analysis.name
            child_data = experiment_data.child_data(child_index)
            bases.append(child_data.metadata['basis'])
            child_result = child_data.analysis_results(result_name)
            ufloat_params = child_result.value.ufloat_params
            for pname, pvalue in ufloat_params.items():
                if pname not in results:
                    results[pname] = np.empty(len(self._analyses), dtype=object)

                results[pname][child_index] = pvalue

        self.options.primary_init = bases[0] // 3
        self._component_results = results

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        # freq
        freqs = unp.nominal_values(self._component_results['freq'])
        user_opt.p0.set_if_empty(freq=np.mean(freqs))

        # Initial and final matrices
        # - set theta = half phase offset @ t=0
        #   (assuming component [0] is t=0)
        # - chi = kappa = 0
        phases = unp.nominal_values(self._component_results['phase'])
        p0 = phases[0] / 2.
        user_opt.p0.set_if_empty(
            theta=p0,
            chi=0.,
            kappa=0.,
        )

        # psi and phi
        amps = unp.nominal_values(self._component_results['amp'])
        args = (user_opt, amps, phases, freqs)
        if self.options.primary_init == 0:
            return HamiltonianTomographyAnalysis.fit_guess_x(*args, set_delta=False)
        if self.options.primary_init == 1:
            return HamiltonianTomographyAnalysis.fit_guess_y(*args, set_delta=False)
        return HamiltonianTomographyAnalysis.fit_guess_z(*args, set_delta=False)


class HamiltonianTomographyScan(BatchExperiment):
    """Batched HamiltonianTomography scanning a variables."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter = None
        options.values = None
        options.max_circuits = 100
        options.dummy_components = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        parameter: str,
        values: Sequence[float],
        rabi_init: Optional[Callable] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        param = schedule.get_parameters(parameter)[0]

        for value in values:
            sched = schedule.assign_parameters({param: value}, inplace=False)
            exp = HamiltonianTomography(physical_qubits, sched, rabi_init=rabi_init,
                                        widths=widths, time_unit=time_unit, backend=backend)
            exp.extra_metadata[parameter] = value
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=None)
                         #analysis=HamiltonianTomographyScanAnalysis(analyses))
        self.set_experiment_options(parameter=parameter, values=values)

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['scan_parameter'] = self.experiment_options.parameter
        return metadata

    @property
    def num_circuits(self) -> int:
        return sum(subexp.num_circuits for subexp in self.component_experiment())

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        if self.experiment_options.dummy_components is not None:
            # [3, scan]
            counts_list = []
            icirc = 0
            for subexp, components in zip(self.component_experiment(),
                                          self.experiment_options.dummy_components.T):
                comp_original = subexp.experiment_options.dummy_components
                subexp.experiment_options.dummy_components = components
                counts_list.extend(
                    subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits])
                )
                subexp.experiment_options.dummy_components = comp_original
                icirc += subexp.num_circuits

            return counts_list

        return super().dummy_data(transpiled_circuits)


class HamiltonianTomographyScanAnalysis(CompoundAnalysis):
    """Analysis for HamiltonianTomographyScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.poly_orders: Optional[tuple[PolynomialOrder, PolynomialOrder, PolynomialOrder]] = None
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        """Fit the components as functions of the scan parameter."""
        component_index = experiment_data.metadata["component_child_index"]

        xvar = experiment_data.metadata['scan_parameter']
        xval = np.empty(len(component_index))
        hamiltonian_components = np.empty((3, len(component_index)), dtype='O')

        for iexp, child_index in enumerate(component_index):
            child_data = experiment_data.child_data(child_index)
            xval[iexp] = child_data.metadata[xvar]
            hamiltonian_components[:, iexp] = \
                child_data.analysis_results('hamiltonian_components').value

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel=xvar,
                ylabel='Hamiltonian component'
            )

            for op, components in zip(['x', 'y', 'z'], hamiltonian_components):
                plotter.set_series_data(
                    op,
                    x_formatted=xval,
                    y_formatted=unp.nominal_values(components),
                    y_formatted_err=unp.std_devs(components)
                )

        if self.options.poly_orders is not None:
            interp_x = np.linspace(xval[0], xval[-1], 100)

            for op, components, poly_order in zip(['x', 'y', 'z'], hamiltonian_components,
                                                  self.options.poly_orders):
                order = PolynomialOrder(poly_order)
                powers = order.powers

                fitfunc = sparse_poly_fitfunc(powers)

                popt, pcov = sciopt.curve_fit(fitfunc, xval, unp.nominal_values(components),
                                              p0=np.zeros_like(powers))

                omega_coeffs = np.full(order.order + 1, ufloat(0., 0.))
                try:
                    omega_coeffs[powers] = correlated_values(nom_values=popt, covariance_mat=pcov)
                except np.linalg.LinAlgError:
                    omega_coeffs[powers] = list(ufloat(v, 0.) for v in popt)

                analysis_results.append(AnalysisResultData(name=f'omega_{op}_coeffs',
                                                           value=omega_coeffs))

                if self.options.plot:
                    plotter.set_series_data(
                        op,
                        x_interp=interp_x,
                        y_interp=fitfunc(interp_x, *popt)
                    )

        if self.options.plot:
            return analysis_results, [plotter.figure()]
        return analysis_results, []
