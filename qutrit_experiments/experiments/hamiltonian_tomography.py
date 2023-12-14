from typing import List, Tuple, Iterable, Optional, Union, Sequence, Callable
import warnings
import logging
from uncertainties import ufloat, correlated_values, unumpy as unp
import numpy as np
from matplotlib.figure import Figure
import scipy.optimize as sciopt
import lmfit

from qiskit import QuantumCircuit
from qiskit.pulse import ScheduleBlock
from qiskit.result import Counts
from qiskit.providers import Backend
from qiskit_experiments.framework import Options, ExperimentData, AnalysisResultData
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..common.framework_overrides import CompoundAnalysis, CompositeAnalysis, BatchExperiment
from ..common.linked_curve_analysis import LinkedCurveAnalysis
from ..common.util import sparse_poly_fitfunc, PolynomialOrder
from .gs_rabi import GSRabi, GSRabiAnalysis
from .bloch import (amp_matrix, amp_func, phase_matrix, phase_func, base_matrix, base_func,
                    unit_bound, pos_unit_bound)
from .dummy_data import single_qubit_counts


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

    For :math:`r(t) := (\langle x \rangle (t), \langle y \rangle (t), \langle z \rangle (t))^T`, we have

    .. math::

        r(t) = e^{G t} r(0)

    where

    .. math::

        G = \begin{pmatrix} 0 & -\omega^z & \omega^y \\ \omega^z & 0 & -\omega^x \\ -\omega^y & \omega^x & 0 \end{pmatrix}.

    Exponentiating this, we have for :math:`r(0) = (0, 0, 1)^T`

    .. math::

        r(t) = \frac{1}{\Omega^2} \begin{pmatrix}
        \omega^x \omega^z [1 - \cos(\Omega t)] + \Omega \omega^y \sin(\Omega t) \\
        \omega^y \omega^z [1 - \cos(\Omega t)] - \Omega \omega^x \sin(\Omega t) \\
        (\omega^z)^2 + [(\omega^x)^2 + (\omega^y)^2] \cos(\Omega t)
        \end{pmatrix},

    where :math:`\Omega := \sqrt{\sum_g (\omega^g)^2}.` Each component of :math:`r(t)` thus evolves sinusoidally
    with the same frequency while satisfying :math:`|r(t)|^2 = 1`. Reparametrizing :math:`\omega^g` as

    .. math::

        \omega^x/\Omega & = \sin \psi \cos \phi, \\
        \omega^y/\Omega & = \sin \psi \sin \phi, \\
        \omega^z\Omega & = \cos \psi,

    we arrive at

    .. math::

        r(t) = \begin{pmatrix}
        \sin \psi [-\cos \psi \cos \phi \cos(\Omega t) + \sin \phi \sin(\Omega t)] + \sin \psi \cos \psi \cos \phi \\
        \sin \psi [-\cos \psi \sin \phi \cos(\Omega t) - \cos \phi \sin(\Omega t)] + \sin \psi \cos \psi \sin \phi \\
        \sin^2 \psi \cos(\Omega t) + \cos^2 \psi
        \end{pmatrix}.

    Casting the right hand side into the standard OscillationAnalysis form :math:`a \cos (2 \pi f t + \Phi) + b`,

    .. math::

        r(t) = \begin{pmatrix}
        \sin \psi A \cos(2 \pi f t + \alpha) + \sin \psi \cos \psi \cos \phi \\
        \sin \psi B \cos(2 \pi f t + \beta) + \sin \psi \cos \psi \sin \phi \\
        \sin^2 \psi \cos(2 \pi f t) + \cos^2 \psi
        \end{pmatrix},

    where

    .. math::

        f & = \Omega / (2 \pi) \\
        A & = \sqrt{\cos^2 \psi \cos^2 \phi + \sin^2 \phi}, \\
        B & = \sqrt{\cos^2 \psi \sin^2 \phi + \cos^2 \phi},

    and

    .. math::

        \cos \alpha & = -\cos \psi \cos \phi / A, \\
        \sin \alpha & = -\sin \phi / A, \\
        \cos \beta & = -\cos \psi \cos \phi / B, \\
        \sin \beta & = \cos \phi / B.
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
        initial_state: Optional[str] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        if rabi_init is None:
            rabi_init = GSRabi

        axes = ['x', 'y', 'z']
        if initial_state is None:
            initial_state = GSRabi._default_experiment_options().initial_state

        bases = [(initial_state, meas_basis) for meas_basis in axes]
        if secondary_trajectory:
            iaxis = axes.index(initial_state)
            orth_axes = [axes[(iaxis + i) % 3] for i in range(1, 3)]
            bases += [(orth_axes[0], meas_basis) for meas_basis in orth_axes]

        for idx, (init, meas_basis) in enumerate(bases):
            exp = rabi_init(physical_qubits, schedule, widths=widths, initial_state=init,
                            meas_basis=meas_basis, time_unit=time_unit, experiment_index=idx,
                            backend=backend)
            # convert init and meas_basis to a numerical value (to be used as fit function arguments)
            exp.extra_metadata['basis'] = axes.index(init) * 3 + axes.index(meas_basis)
            if not backend.simulator:
                exp.set_experiment_options(invert_x_sign=True)
            if measured_logical_qubit is not None:
                exp.set_experiment_options(measured_logical_qubit=measured_logical_qubit)

            # Bound the frequency to be unsigned. To allow the fit value of 0 comfortably, the lower
            # bound is set slightly negative.
            exp.analysis.options.bounds['freq'] = (-1.e+3, np.inf)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=HamiltonianTomographyAnalysis(analyses))

        self.extra_metadata = {
            'initial_state': initial_state,
            'secondary_trajectory': secondary_trajectory
        }

    @property
    def num_circuits(self) -> int:
        return sum(subexp.num_circuits for subexp in self.component_experiment())

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[Counts]:
        hamiltonian_components = self.experiment_options.dummy_components
        if hamiltonian_components is None:
            hamiltonian_components = np.array([-1857082.,  -1232100., -138360.])

        paulis = np.array([[[0., 1.], [1., 0.]],
                           [[0., -1.j], [1.j, 0.]],
                           [[1., 0.], [0., -1.]]])
        hamiltonian = np.tensordot(hamiltonian_components, paulis, (0, 0))

        counts_list = []
        icirc = 0

        for imeas in range(3):
            subexp = self.component_experiment(imeas)
            options = subexp.experiment_options
            dummy_components = options.dummy_components
            options.dummy_components = hamiltonian_components
            counts_list.extend(subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits]))
            options.dummy_components = dummy_components
            icirc += subexp.num_circuits

        return counts_list


class HamiltonianTomographyAnalysis(LinkedCurveAnalysis):
    """Track the state along a circle on the surface of the Bloch sphere."""
    @staticmethod
    def fit_guess_x(
        user_opt: curve.FitOptions,
        indiv_amp: np.ndarray,
        indiv_phase: np.ndarray,
        indiv_freq: np.ndarray,
        x_index: int = 0,
        set_delta: bool = True
    ):
        """Provide initial parameter values for initial state on the X axis."""
        y_index = (x_index + 1) % 3
        spsi2cphi2 = 1. - pos_unit_bound(indiv_amp[x_index])
        if spsi2cphi2 > 0.99 and len(indiv_amp) > 3: # rotation about x axis
            # indiv_*[3:] are measured in y & z axes
            return HamiltonianTomographyAnalysis.fit_guess_y(user_opt, indiv_amp[3:],
                                                             indiv_phase[3:], indiv_freq[3:],
                                                             y_index=0, set_delta=set_delta)

        delta = indiv_phase[x_index]
        gamma = indiv_phase[y_index] - delta
        C = indiv_amp[y_index]
        cpsi = -unit_bound(np.sin(gamma) * C)
        psi = np.arccos(cpsi)
        abscphi = pos_unit_bound(np.sqrt(spsi2cphi2 / (1. - (cpsi ** 2))))

        if set_delta:
            user_opt.p0.set_if_empty(
                delta=delta
            )
        user_opt.p0.set_if_empty(
            psi=psi,
            freq=indiv_freq[x_index]
        )

        options = []

        phi_candidates = [np.arccos(abscphi), np.arccos(-abscphi),
                          -np.arccos(-abscphi), -np.arccos(abscphi)]
        for phi in phi_candidates:
            opt = user_opt.copy()
            opt.p0.set_if_empty(phi=phi)
            options.append(opt)

        return options

    @staticmethod
    def fit_guess_y(
        user_opt: curve.FitOptions,
        indiv_amp: np.ndarray,
        indiv_phase: np.ndarray,
        indiv_freq: np.ndarray,
        y_index: int = 1,
        set_delta: bool = True
    ):
        """Provide initial parameter values for initial state on the Y axis."""
        z_index = (y_index + 1) % 3
        spsi2sphi2 = 1. - pos_unit_bound(indiv_amp[y_index])
        if spsi2sphi2 > 0.99 and len(indiv_amp) > 3: # rotation about y axis
            return HamiltonianTomographyAnalysis.fit_guess_z(user_opt, indiv_amp[3:],
                                                             indiv_phase[3:], indiv_freq[3:],
                                                             z_index=0, set_delta=set_delta)

        delta = indiv_phase[y_index]
        beta = delta - indiv_phase[z_index]
        B = indiv_amp[z_index]
        spsicphi = unit_bound(np.sin(beta) * B)
        spsi = pos_unit_bound(np.sqrt(spsicphi ** 2 + spsi2sphi2))

        if set_delta:
            user_opt.p0.set_if_empty(
                delta=delta
            )
        user_opt.p0.set_if_empty(
            freq=indiv_freq[y_index]
        )

        options = []

        if spsi == 1.:
            user_opt.p0.set_if_empty(
                psi=np.pi / 2.
            )
            phi_candidates = [np.arccos(spsicphi), -np.arccos(spsicphi)]
            for phi in phi_candidates:
                opt = user_opt.copy()
                opt.p0.set_if_empty(phi=phi)
                options.append(opt)
        else:
            psi_candidates = [np.arcsin(spsi), np.pi - np.arcsin(spsi)]
            for psi in psi_candidates:
                phi = np.arctan2(-np.cos(beta) / np.cos(psi), np.sin(beta))
                opt = user_opt.copy()
                opt.p0.set_if_empty(psi=psi, phi=phi)
                options.append(opt)

        return options

    @staticmethod
    def fit_guess_z(
        user_opt: curve.FitOptions,
        indiv_amp: np.ndarray,
        indiv_phase: np.ndarray,
        indiv_freq: np.ndarray,
        z_index: int = 2,
        set_delta: bool = True
    ):
        """Provide initial parameter values for initial state on the Z axis."""
        x_index = (z_index + 1) % 3
        spsi2 = pos_unit_bound(indiv_amp[z_index])
        if spsi2 < 0.01 and len(indiv_amp) > 3: # rotation about y axis
            return HamiltonianTomographyAnalysis.fit_guess_x(user_opt, indiv_amp[3:],
                                                             indiv_phase[3:], indiv_freq[3:],
                                                             x_index=0, set_delta=set_delta)
        spsi = np.sqrt(spsi2)
        delta = indiv_phase[z_index]
        alpha = indiv_phase[x_index] - delta
        A = indiv_amp[x_index]
        sphi = unit_bound(-np.sin(alpha) * A / spsi)

        if set_delta:
            user_opt.p0.set_if_empty(
                delta=delta
            )
        user_opt.p0.set_if_empty(
            freq=indiv_freq[z_index]
        )

        options = []

        if spsi == 1.:
            user_opt.p0.set_if_empty(
                psi=np.pi / 2.
            )
            phi_candidates = [np.arcsin(sphi), np.pi - np.arcsin(sphi)]
            for phi in phi_candidates:
                opt = user_opt.copy()
                opt.p0.set_if_empty(phi=phi)
                options.append(opt)
        else:
            psi_candidates = [np.arcsin(spsi), np.pi - np.arcsin(spsi)]
            for psi in psi_candidates:
                phi = np.arctan2(-np.sin(alpha), -np.cos(alpha) / np.cos(psi))
                opt = user_opt.copy()
                opt.p0.set_if_empty(psi=psi, phi=phi)
                options.append(opt)

        return options

    def __init__(self, analyses: List[GSRabiAnalysis]):
        super().__init__(
            analyses,
            linked_params={
                'amp': lmfit.Model(amp_func),
                'freq': None,
                'phase': lmfit.Model(phase_func),
                'base': lmfit.Model(base_func)
            },
            experiment_params=['basis'],
            labels=['x', 'y', 'z', 'o1', 'o2'] # o1 and o2 are not used if secondary_trajectory=False
        )

        self.set_options(
            bounds={
                'psi': (-1.e-3, np.pi + 1.e-3),
                'phi': (-twopi, twopi),
                'delta': (-np.pi, 3. * np.pi),
                'freq': (-1.e+5, np.inf) # giving some slack on the negative side
            }
        )
        self.plotter.set_figure_options(
            ylim=(-1.1, 1.1)
        )

        self._initial_state = ''

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.
        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted` data collection to fit.
        Returns:
            List of fit options that are passed to the fitter function.

        TODO write the analysis for when azimuthal projections are measured
        """
        ## Initial values for the linked params
        indiv_amp = unp.nominal_values(self._individual_fit_results['amp'])
        indiv_phase = unp.nominal_values(self._individual_fit_results['phase'])
        indiv_freq = unp.nominal_values(self._individual_fit_results['freq'])

        if self._initial_state == 'x':
            return self.fit_guess_x(user_opt, indiv_amp, indiv_phase, indiv_freq)
        elif self._initial_state == 'y':
            return self.fit_guess_y(user_opt, indiv_amp, indiv_phase, indiv_freq)
        else:
            return self.fit_guess_z(user_opt, indiv_amp, indiv_phase, indiv_freq)

    def _run_curve_fit(
        self,
        curve_data: curve.CurveData,
        models: List[lmfit.Model],
    ) -> curve.CurveFitResult:
        result = super()._run_curve_fit(curve_data, models)

        if result.success:
            while result.params['phi'] < -np.pi:
                result.params['phi'] += twopi
            while result.params['phi'] > np.pi:
                result.params['phi'] -= twopi
            while result.params['delta'] < 0.:
                result.params['delta'] += twopi
            while result.params['delta'] > twopi:
                result.params['delta'] -= twopi

        return result

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Compute the Hamiltonian components."""
        self._initial_state = experiment_data.metadata['initial_state']
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)
        fit_result = next(res.value for res in analysis_results
                          if res.name == PARAMS_ENTRY_PREFIX + self.name)
        popt = fit_result.ufloat_params

        omega = popt['freq'] * twopi

        # Divide by factor two to interpret as coefficients of single-qubit X, Y, Z
        components = np.array([
            omega * unp.sin(popt['psi']) * unp.cos(popt['phi']),
            omega * unp.sin(popt['psi']) * unp.sin(popt['phi']),
            omega * unp.cos(popt['psi'])
        ]) / 2.

        analysis_results.append(AnalysisResultData(name='hamiltonian_components', value=components))

        return analysis_results, figures


class HamiltonianTomographyAnalysisNew(CompoundAnalysis, curve.CurveAnalysis):
    @staticmethod
    def evolution_factory(init, meas):
        def evolution(x, freq, psi, phi, theta, chi, kappa):
            xdims = tuple(range(len(np.asarray(x).shape)))
            x = np.expand_dims(x, (-2, -1))

            amp = np.expand_dims(amp_matrix(chi, kappa), xdims)
            phase = np.expand_dims(phase_matrix(chi, kappa, 0.), xdims)
            base = np.expand_dims(base_matrix(chi, kappa), xdims)
            matrix = (amp * np.cos(theta + phase) + base)

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

    def __init__(self, analyses: List[GSRabiAnalysis]):
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
        for idx, analysis in enumerate(self._analyses):
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
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
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

        self._primary_init = bases[0] // 3
        self._secondary_init = None if len(bases) == 3 else bases[3] // 3
        self._component_results = results

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
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
        if self._primary_init == 0:
            return HamiltonianTomographyAnalysis.fit_guess_x(*args, set_delta=False)
        elif self._primary_init == 1:
            return HamiltonianTomographyAnalysis.fit_guess_y(*args, set_delta=False)
        elif self._primary_init == 2:
            return HamiltonianTomographyAnalysis.fit_guess_z(*args, set_delta=False)


class HamiltonianTomographyScan(BatchExperiment):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.values = None
        options.max_circuits = 100
        options.dummy_components = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        parameter: Union[str, Tuple[str, ...]],
        values: Union[Sequence[float], Tuple[Sequence[float], ...]],
        rabi_init: Optional[Callable] = None,
        widths: Optional[Iterable[float]] = None,
        initial_state: Optional[str] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        if isinstance(parameter, str):
            parameter = (parameter,)
            values = (values,)

        experiments = []
        analyses = []

        params = tuple(schedule.get_parameters(name)[0] for name in parameter)

        for value_tuple in zip(*values):
            assign_parameters = dict(zip(params, value_tuple))
            sched = schedule.assign_parameters(assign_parameters, inplace=False)
            exp = HamiltonianTomography(physical_qubits, sched, rabi_init=rabi_init,
                                        widths=widths, initial_state=initial_state,
                                        secondary_trajectory=secondary_trajectory,
                                        time_unit=time_unit, backend=backend)
            for name, value in zip(parameter, value_tuple):
                exp.extra_metadata[name] = value

            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = HamiltonianTomographyScanAnalysis(analyses, parameter[0])

        super().__init__(experiments, backend=backend, analysis=analysis)

        self.set_experiment_options(values=values)

    @property
    def num_circuits(self) -> int:
        return sum(subexp.num_circuits for subexp in self.component_experiment())

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[Counts]:
        if self.experiment_options.dummy_components is not None:
            # [3, scan]
            counts_list = []
            icirc = 0
            for subexp, components in zip(self.component_experiment(),
                                          self.experiment_options.dummy_components.T):
                comp_original = subexp.experiment_options.dummy_components
                subexp.experiment_options.dummy_components = components
                counts_list.extend(subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits]))
                subexp.experiment_options.dummy_components = comp_original
                icirc += subexp.num_circuits

            return counts_list

        return super().dummy_data(transpiled_circuits)


class HamiltonianTomographyScanAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.poly_orders = None

        return options

    def __init__(
        self,
        analyses: List[HamiltonianTomographyAnalysis],
        xvar: str,
        poly_orders: Optional[Tuple[PolynomialOrder, PolynomialOrder, PolynomialOrder]] = None
    ):
        super().__init__(analyses)
        self.xvar = xvar
        self.set_options(poly_orders=poly_orders)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Fit the components as functions of the scan parameter."""
        component_index = experiment_data.metadata["component_child_index"]

        xval = np.empty(len(component_index))
        hamiltonian_components = np.empty((3, len(component_index)), dtype='O')

        for iexp, child_index in enumerate(component_index):
            child_data = experiment_data.child_data(child_index)
            xval[iexp] = child_data.metadata[self.xvar]
            hamiltonian_components[:, iexp] = child_data.analysis_results('hamiltonian_components').value

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel=self.xvar,
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

            for op, components, poly_order in zip(['x', 'y', 'z'], hamiltonian_components, self.options.poly_orders):
                order = PolynomialOrder(poly_order)
                powers = order.powers

                fitfunc = sparse_poly_fitfunc(powers)

                popt, pcov = sciopt.curve_fit(fitfunc, xval, unp.nominal_values(components), p0=np.zeros_like(powers))

                omega_coeffs = np.full(order.order + 1, ufloat(0., 0.))
                try:
                    omega_coeffs[powers] = correlated_values(nom_values=popt, covariance_mat=pcov)
                except np.linalg.LinAlgError:
                    omega_coeffs[powers] = list(ufloat(v, 0.) for v in popt)

                analysis_results.append(AnalysisResultData(name=f'omega_{op}_coeffs', value=omega_coeffs))

                if self.options.plot:
                    plotter.set_series_data(
                        op,
                        x_interp=interp_x,
                        y_interp=fitfunc(interp_x, *popt)
                    )

        if self.options.plot:
            return analysis_results, [plotter.figure()]
        else:
            return analysis_results, []
