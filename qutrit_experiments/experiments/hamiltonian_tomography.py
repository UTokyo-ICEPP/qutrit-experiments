"""Hamiltonian tomography experiment."""
from collections.abc import Callable, Iterable, Sequence
from itertools import product
import logging
from typing import Any, Optional, Union
from matplotlib.figure import Figure
import lmfit
import numpy as np
from scipy.optimize import curve_fit, least_squares
from uncertainties import ufloat, correlated_values, unumpy as unp

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.result import Counts
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import Options, ExperimentData, AnalysisResultData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from ..util.bloch import so3_polar, unit_bound, pos_unit_bound
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

    where

    .. math::

        U_i & = \lim_{\substack{\Delta t \to 0 \\ N\Delta t = \tau}}
              \prod_{j=1}^{N} e^{-iH(j\Delta t)\Delta t} \\
            & = T \left[ e^{-i\int_{0}^{\tau} H(t) dt} \right] \\
        U_f & = \lim_{\substack{\Delta t \to 0 \\ N\Delta t = \tau}}
              \prod_{j=1}^{N} e^{-iH(T - (N-j)\Delta t)\Delta t} \\
            & = T \left[ e^{-i\int_{0}^{\tau} H(T - \tau + t)} \right] \\

    with :math:`\tau` the rise and fall time and :math:`T` the overall pulse duration. The product
    symbol implies left multiplication of operators. Assuming :math:`H(T - t) = H(t)`, we have

    .. math::

        U_f & = \lim_{\substack{\Delta t \to 0 \\ N\Delta t = \tau}}
              \prod_{j=1}^{N} e^{-iH((N-j)\Delta t)\Delta t} \\
            & = \lim_{\substack{\Delta t \to 0 \\ N\Delta t = \tau}}
              {\prod^*}_{j=0}^{N-1} e^{-iH(j\Delta t)\Delta t} \\
            & = \left[\lim_{\substack{\Delta t \to 0 \\ N\Delta t = \tau}}
              \prod_{j=1}^{N} e^{-iH^T(j\Delta t)\Delta t}\right]^T \\

    where :math:`\prod^*` represents right multiplication. With the pauli decomposition of
    :math:`H` as in the first equation,

    .. math::

        H^T = \frac{\omega^x}{2} x - \frac{\omega^y}{2} y + \frac{\omega^z}{2} z.

    If :math:`\omega^y \ll \omega^x, \omega^z` during the rise and fall evolutions,

    .. math::

        U_f = U_i^T

    Then, if the Bloch representation of :math:`U_i` is given by a three-dimentional rotation matrix
    with angle :math:`\theta` and axis :math:`(\sin\chi\cos\kappa, \sin\chi\sin\kappa, \cos\chi)`,
    the Bloch representation of :math:`U_f` has the same angle and axis where
    :math:`\kappa \to -\kappa`.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        rabi_init: Optional[Callable] = None,
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
            exp.extra_metadata['init'] = init
            exp.extra_metadata['meas_basis'] = meas_basis
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


class HamiltonianTomographyAnalysis(CompoundAnalysis, curve.CurveAnalysis):
    """Analysis for HamiltonianTomography."""
    @staticmethod
    def zx_evolution_factory(init, meas_basis):
        i_init = ['x', 'y', 'z'].index(init)
        i_meas = ['x', 'y', 'z'].index(meas_basis)
        def evolution(x, omega, psi, phi, theta, chi, kappa):
            xdims = tuple(range(np.asarray(x).ndim))
            theta = np.expand_dims(theta, xdims)
            return np.einsum('...i,...ij,...j->...',
                             so3_polar(theta, chi, -kappa)[..., i_meas, :],
                             so3_polar(omega * x, psi, phi),
                             so3_polar(theta, chi, kappa)[..., i_init])
        return evolution

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.zx_evolution = True
        return options

    def __init__(self, analyses: list[GSRabiAnalysis]):
        super().__init__(analyses, flatten_results=False)
        curve.CurveAnalysis.__init__(self, [])

        self.set_options(
            data_processor=DataProcessor('counts', [Probability('1'), BasisExpectationValue()]),
            data_subfit_map={
                f'{meas_basis}|{init}': {'initial_state': init, 'meas_basis': meas_basis}
                for init in ['z', 'x'] for meas_basis in ['x', 'y', 'z']
            },
            bounds={
                'psi': (-1.e-3, np.pi + 1.e-3),
                'phi': (-twopi, twopi),
                'omega': (-1.e+5, np.inf), # giving some slack on the negative side
            }
        )
        self.plotter.set_figure_options(
            ylim=(-1.1, 1.1)
        )

        # Fit results of individual components
        self._component_results = {}

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = curve.CurveAnalysis._run_analysis(self, experiment_data)

        fit_result = next(res.value for res in analysis_results
                          if res.name == PARAMS_ENTRY_PREFIX + self.name)
        popt = fit_result.ufloat_params
        # Divide the omegas by factor two to interpret as coefficients of single-qubit X, Y, Z
        analysis_results.append(
            AnalysisResultData(
                name='hamiltonian_components',
                value=np.array([
                    popt['omega'] * unp.sin(popt['psi']) * unp.cos(popt['phi']), # pylint: disable=no-member
                    popt['omega'] * unp.sin(popt['psi']) * unp.sin(popt['phi']), # pylint: disable=no-member
                    popt['omega'] * unp.cos(popt['psi']) # pylint: disable=no-member
                ]) / 2.
            )
        )
        return analysis_results, figures

    def _initialize(
        self,
        experiment_data: ExperimentData,
    ):
        # U_i and U_f parameters are frozen and determined from a fit to t=0 data points
        if self.options.zx_evolution:
            self._models = [lmfit.Model(self.zx_evolution_factory(init, meas_basis),
                                        name=f'{meas_basis}|{init}')
                            for init in ['z', 'x'] for meas_basis in ['x', 'y', 'z']]
            self.set_options(
                fixed_parameters={
                    'theta': 0.,
                    'chi': 0.,
                    'kappa': 0.
                }
            )
        else:
            raise NotImplementedError('non-zx')

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

        component_index = experiment_data.metadata['component_child_index']
        for analysis, child_index in zip(self._analyses, component_index):
            child_data = experiment_data.child_data(child_index)
            child_result = child_data.analysis_results(PARAMS_ENTRY_PREFIX + analysis.name)
            init = child_data.metadata['init']
            meas_basis = child_data.metadata['meas_basis']
            self._component_results[f'{meas_basis}|{init}'] = child_result.value.ufloat_params

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        subdata = {}
        reliable_results = {}
        for label in curve_data.labels:
            subdata[label] = curve_data.get_subset_of(label)
            if not np.isclose(subdata[label].x[0], 0.):
                raise AnalysisError('First x value must be 0')
            if np.nanmax(subdata[label].y) - np.nanmin(subdata[label].y) > 0.5:
                reliable_results[label] = self._component_results[label]
        if not reliable_results:
            # If no reliable points are found, just use all
            logger.warning('No init x meas_basis setup had y value range greater than 0.5')
            reliable_results = self._component_results

        # Fit to U_f U_i
        y_0 = np.array([subdata[model._name].y[0] for model in self._models])
        if self.options.zx_evolution:
            def set_params(params):
                return {'theta': params[0], 'chi': params[1], 'kappa': params[2]}
            p0 = (0.01, np.pi / 2., 0.)
            bounds = ([-twopi, 0., -np.pi], [twopi, np.pi, np.pi])
        else:
            raise NotImplementedError('non-zx')

        def residual(params):
            y_pred = np.array([model.eval(x=0., omega=0., psi=0., phi=0., **set_params(params))
                               for model in self._models])
            return y_pred - y_0

        res = least_squares(residual, p0, bounds=bounds)
        user_opt.p0.update(set_params(res.x))

        # freq is common but reliable only when amp is sufficiently large
        omega_p0 = twopi * np.mean([res['freq'].n for res in reliable_results.values()])
        logger.debug('Initial guess for omega=%f from %s', omega_p0,
                     {key: value['freq'] for key, value in reliable_results.items()})
        user_opt.p0.set_if_empty(omega=omega_p0)

        # Guess from init=Z (assuming no contribution from rise & fall)
        def estimate_amp(label):
            xdata = subdata[label].x
            ydata = subdata[label].y
            popt, _ = curve_fit(
                lambda x, amp, base: amp * np.cos(omega_p0 * x) + base,
                xdata, ydata, (0., ydata[0])
            )
            return unit_bound(np.abs(popt[0]))

        try:
            spsi2 = pos_unit_bound(reliable_results['z|z']['amp'].n)
        except KeyError:
            spsi2 = estimate_amp('z|z')
        logger.debug('(sin psi)^2 = %f', spsi2)
        # B^2 - A^2 = spsi2 * (-cpsi2 * c2phi + c2phi) = spsi4 * c2phi
        try:
            A = pos_unit_bound(reliable_results['x|z']['amp'].n) # pylint: disable=invalid-name
        except KeyError:
            A = estimate_amp('x|z') # pylint: disable=invalid-name
        try:
            B = pos_unit_bound(reliable_results['y|z']['amp'].n) # pylint: disable=invalid-name
        except KeyError:
            B = estimate_amp('y|z')
        c2phi = unit_bound((B ** 2 - A ** 2) / spsi2 ** 2)
        logger.debug('A = %f, B = %f, cos 2phi = %f', A, B, c2phi)
        options = []
        # For phi = delta + n * pi/2 (0 < delta < pi/2, n=0,1,2,3)
        # cos(2phi) = cos(2*delta + n*pi) = (-1)^n cos(2*delta)
        # arccos(cos(2phi)) = 2*delta (n=0,2)
        #                   = pi - 2*delta (n=1,3)
        for phi_quadrant in range(4):
            if phi_quadrant % 2 == 0:
                phi = np.arccos(c2phi) / 2.
            else:
                phi = (np.pi - np.arccos(c2phi)) / 2.
            phi += np.pi / 2. * phi_quadrant
            logger.debug('phi = %f (n=%d)', phi, phi_quadrant)
            opt = user_opt.copy()
            opt.p0.set_if_empty(
                psi=np.arcsin(np.sqrt(spsi2)),
                phi=phi
            )
            options.append(opt)

        return options


class HamiltonianTomographyScan(BatchExperiment):
    """Batched HamiltonianTomography scanning a variable."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter = None
        options.values = None
        options.max_circuits = 100
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

        super().__init__(experiments, backend=backend,
                         analysis=HamiltonianTomographyScanAnalysis(analyses))
        self.set_experiment_options(parameter=parameter, values=values)

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['scan_parameter'] = self.experiment_options.parameter
        return metadata

    @property
    def num_circuits(self) -> int:
        return sum(subexp.num_circuits for subexp in self.component_experiment())


class HamiltonianTomographyScanAnalysis(CompoundAnalysis):
    """Analysis for HamiltonianTomographyScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        # (PolynomialOrder, PolynomialOrder, PolynomialOrder)
        options.poly_orders = None
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
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
                    #y_formatted_err=unp.std_devs(components)
                    y_formatted_err=np.full_like(components, 100., dtype=float)
                )

        if self.options.poly_orders is not None:
            interp_x = np.linspace(xval[0], xval[-1], 100)

            for op, components, poly_order in zip(['x', 'y', 'z'], hamiltonian_components,
                                                  self.options.poly_orders):
                order = PolynomialOrder(poly_order)
                powers = order.powers

                fitfunc = sparse_poly_fitfunc(powers)

                popt, pcov = curve_fit(fitfunc, xval, unp.nominal_values(components),
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
