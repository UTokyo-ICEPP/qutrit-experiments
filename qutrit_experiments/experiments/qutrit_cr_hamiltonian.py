r"""Identify the Bloch-space rotation matrix of the target qubit under a CR tone.

The effective Hamiltonian of a CR drive on a qutrit-qubit pair is

.. math::

    H = \frac{I \otimes \sum_g \omega_{Ig} g}{2} + \frac{z \otimes \sum_g \omega_{zg} g}{2}
        + \frac{\zeta \otimes \sum_g \omega_{\zeta g}}{2}

where $g = x, y, z$.

When the control qutrit state is fixed, the Hamiltonians on the target qubit are

.. math::

    H^0 & = \sum_g \frac{\omega_{Ig} + \omega_{zg}}{2} g = \sum_g \frac{\omega^0_g}{2} g \\
    H^1 & = \sum_g \frac{\omega_{Ig} - \omega_{zg} + \omega_{\zeta g}}{2} g
          = \sum_g \frac{\omega^1_g}{2} g \\
    H^2 & = \sum_g \frac{\omega_{Ig} - \omega_{\zeta g}}{2} g = \sum_g \frac{\omega^2_g}{2} g.

The evolution of the expectation value of $g$ is given by

.. math::

    \frac{d}{dt} \langle g \rangle^k (t) = i\langle [H^k, g] \rangle.

For :math:`r^k(t) := (\langle x \rangle^k (t), \langle y \rangle^k (t), \langle z \rangle^k (t))^T`,
we have

.. math::

    r^k(t) = e^{G^k t} r^k(0)

where

.. math::

    G^k = \begin{pmatrix}
        0 & -\omega^k_z & \omega^k_y \\
        \omega^k_z & 0 & -\omega^k_x \\
        -\omega^k_y & \omega^k_x & 0
    \end{pmatrix}.

Exponentiating this, we have for :math:`r^k(0) = (0, 0, 1)^T`

.. math::

    r^k(t) = \frac{1}{(\Omega^{k})^2} \begin{pmatrix}
        \omega^k_x \omega^k_z [1 - \cos(\Omega^k t)] + \Omega^k \omega^k_y \sin(\Omega^k t) \\
        \omega^k_y \omega^k_z [1 - \cos(\Omega^k t)] - \Omega^k \omega^k_x \sin(\Omega^k t) \\
        (\omega^k_z)^2 + [(\omega^k_x)^2 + (\omega^k_y)^2] \cos(\Omega t)
    \end{pmatrix},

where :math:`\Omega^k := \sqrt{\sum_g (\omega^k_g)^2}`. Each component of :math:`r^k(t)` thus
evolves sinusoidally with the same frequency while satisfying :math:`|r^k(t)|^2 = 1`.
Reparametrizing :math:`\omega^k_g` as

.. math::

    \omega^k_x/\Omega^k & = \sin \psi^k \cos \phi^k, \\
    \omega^k_y/\Omega^k & = \sin \psi^k \sin \phi^k, \\
    \omega^k_z/\Omega^k & = \cos \psi^k,

we arrive at

.. math::

    r^k(t) = \begin{pmatrix}
        \sin \psi^k [-\cos \psi^k \cos \phi^k \cos(\Omega^k t) + \sin \phi^k \sin(\Omega^k t)]
            + \sin \psi^k \cos \psi^k \cos \phi^k \\
        \sin \psi^k [-\cos \psi^k \sin \phi^k \cos(\Omega^k t) - \cos \phi^k \sin(\Omega^k t)]
            + \sin \psi^k \cos \psi^k \sin \phi^k \\
        \sin^2 \psi^k \cos(\Omega^k t) + \cos^2 \psi^k
    \end{pmatrix}.

Casting the right hand side into the standard OscillationAnalysis form
:math:`a \cos (2 \pi f t + \Phi) + b`,

.. math::

    r^k(t) = \begin{pmatrix}
        \sin \psi^k A^k \cos(2 \pi f^k t + \alpha^k) + \sin \psi^k \cos \psi^k \cos \phi^k \\
        \sin \psi^k B^k \cos(2 \pi f^k t + \beta^k) + \sin \psi^k \cos \psi^k \sin \phi^k \\
        \sin^2 \psi^k \cos(2 \pi f^k t) + \cos^2 \psi^k
    \end{pmatrix},

where

.. math::

    f^k & = \Omega^k / (2 \pi) \\
    A^k & = \sqrt{\cos^2 \psi^k \cos^2 \phi^k + \sin^2 \phi^k}, \\
    B^k & = \sqrt{\cos^2 \psi^k \sin^2 \phi^k + \cos^2 \phi^k},

and

.. math::

    \cos \alpha^k & = -\cos \psi^k \cos \phi^k / A^k, \\
    \sin \alpha^k & = -\sin \phi^k / A^k, \\
    \cos \beta^k & = -\cos \psi^k \cos \phi^k / B^k, \\
    \sin \beta^k & = \cos \phi^k / B^k.

"""
from collections.abc import Iterable, Sequence
from typing import Optional, Union
from matplotlib.figure import Figure
import numpy as np
import numpy.polynomial as poly
import scipy.optimize as sciopt
from uncertainties import ufloat, correlated_values, unumpy as unp

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.result import Counts
from qiskit_experiments.framework import Options, BaseAnalysis, ExperimentData, AnalysisResultData
from qiskit_experiments.framework.matplotlib import default_figure_canvas
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from ..util.polynomial import PolynomialOrder, sparse_poly_fitfunc
from .cr_rabi import cr_rabi_init
from .hamiltonian_tomography import HamiltonianTomography, HamiltonianTomographyScan

twopi = 2. * np.pi


class QutritCRHamiltonian(BatchExperiment):
    """Hamiltonian tomography of qutrit-qubit CR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.dummy_components = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        widths: Optional[Iterable[float]] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for control_state in range(3):
            exp = HamiltonianTomography(physical_qubits, schedule,
                                        rabi_init=cr_rabi_init(control_state),
                                        widths=widths, secondary_trajectory=secondary_trajectory,
                                        time_unit=time_unit, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = QutritCRHamiltonianAnalysis(analyses)

        super().__init__(experiments, backend=backend, analysis=analysis)

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        # Coefficients of [[Ix/2, Iy/2, Iz/2], [zx/2, zy/2, zz/2], [ζx/2, ζy/2, ζz/2]]
        hamiltonian_components = self.experiment_options.dummy_components
        if hamiltonian_components is None:
            hamiltonian_components = np.array([
                [-3916720.77857143, -2662369.37857143, 1096392.2505241197],
                [-5.48340909e+06, -3.72731713e+06,  6.14156360e+05],
                [5.49052361e+06, 4.05421662e+06, 1.12147049e+06]
            ])

       # control_basis_components[c, i]
       #               = <c|I|c>*nu_Ii + <c|z|c>*nu_zi + <c|ζ|c>*nu_ζi  (c=0,1,2; i=x,y,z)
       #   for c=0,1,2:   [1,1,1]        [1,-1,0]         [0,1,-1]
       # control_eigvals = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) # [c, I/z/ζ]
       # state_components = (control_eigvals @ hamiltonian_components) / 2.

        # A calculation of state components independent from how it's done in the analysis
        paulis = np.array([[[0., 1.], [1., 0.]],
                           [[0., -1.j], [1.j, 0.]],
                           [[1., 0.], [0., -1.]]])
        control_ops = np.array([np.eye(3), np.diagflat([1, -1, 0]), np.diagflat([0, 1, -1])],
                               dtype=complex)
        hamiltonian = np.einsum('cij,tkl,ct->ikjl',
                                control_ops / 2., paulis, hamiltonian_components)
        # Take the trace / 2 with Paulis and extract the diagonal blocks
        state_components = np.einsum('ikjl,blk->ijb',
                                     hamiltonian,
                                     paulis)[[0, 1, 2], [0, 1, 2]] / 2.

        counts_list = []
        icirc = 0

        for control_state in range(3):
            subexp = self.component_experiment(control_state)
            options = subexp.experiment_options
            dummy_components = options.dummy_components
            options.dummy_components = state_components[control_state]
            counts_list.extend(
                subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits])
            )
            options.dummy_components = dummy_components
            icirc += subexp.num_circuits

        return counts_list


class QutritCRHamiltonianAnalysis(CompoundAnalysis):
    """Hamiltonian tomography analysis for CR tone with a control qutrit."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def __init__(self, analyses: list[BaseAnalysis]):
        super().__init__(analyses)

        self.figure = Figure(figsize=[9.6, 4.8])
        _ = default_figure_canvas(self.figure)
        axs = self.figure.subplots(3, 1, sharex=True)
        for control_state, (analysis, ax) in enumerate(zip(analyses, axs)):
            analysis.plotter.set_options(axis=ax)
            analysis.plotter.set_figure_options(
                figure_title=fr'Control: $|{control_state}\rangle$'
            )

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        for analysis in self._analyses:
            analysis.options.plot = self.options.plot

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        """Compute the Hamiltonian components from the fit results."""
        control_basis_components = np.empty((3, 3), dtype=object)

        component_index = experiment_data.metadata["component_child_index"]

        for control_state, in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            # omega^c_g/2
            control_basis_components[control_state] = \
                child_data.analysis_results('hamiltonian_components').value

        control_eigvals = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) # [c, I/z/ζ]
        # Multiply by factor two to obtain omega_[Izζ]
        components = (np.linalg.inv(control_eigvals) @ control_basis_components) * 2

        analysis_results.append(
            AnalysisResultData(
                name='hamiltonian_components',
                value=components
            )
        )

        if all(subanalysis.options.plot for subanalysis in self._analyses):
            figures.append(self.figure)

        return analysis_results, figures


class QutritCRHamiltonianScan(BatchExperiment):
    """Batched QutritCRHamiltonian scanning one or more parameters."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.dummy_components = None
        return options

    @classmethod
    def _default_transpile_options(cls) -> Options:
        return HamiltonianTomographyScan._default_transpile_options()

    def __init__(
        self,
        physical_qubits: tuple[int, int],
        schedule: ScheduleBlock,
        parameter: Union[str, tuple[str, ...]],
        values: Union[Sequence[float], tuple[Sequence[float], ...]],
        widths: Optional[Iterable[float]] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for control_state in range(3):
            exp = HamiltonianTomographyScan(physical_qubits, schedule, parameter, values,
                                            rabi_init=cr_rabi_init(control_state), widths=widths,
                                            secondary_trajectory=secondary_trajectory,
                                            time_unit=time_unit, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = QutritCRHamiltonianScanAnalysis(analyses)

        super().__init__(experiments, backend=backend, analysis=analysis)

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        if self.experiment_options.dummy_components is not None:
            # [control, target, scan]
            control_eigvals = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) # [c, I/z/ζ]
            state_components = np.einsum('ij,jks->iks', control_eigvals,
                                         self.experiment_options.dummy_components) / 2.

            counts_list = []
            icirc = 0
            for control_state, subexp in enumerate(self.component_experiment()):
                options = subexp.experiment_options
                orig_comp = options.dummy_components
                options.dummy_components = state_components[control_state]
                counts_list.extend(
                    subexp.dummy_data(transpiled_circuits[icirc:icirc + subexp.num_circuits])
                )
                options.dummy_components = orig_comp
                icirc += subexp.num_circuits

            return counts_list

        return super().dummy_data(transpiled_circuits)


class QutritCRHamiltonianScanAnalysis(CompoundAnalysis):
    """Analysis for QutritCRHamiltonianScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.plot_angle = False
        # If poly_orders is set here, fits with respect to control_op will be performed
        # {(ic, ib): order}
        options.poly_orders = {}

        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        """Linearly transform the c=0, 1, 2 scan results to c=I, z, ζ."""
        xvar = ''
        xval = None
        poly_orders = None
        control_basis_components = []
        control_basis_coeffs = []

        control_ops = ['I', 'z', 'ζ']
        target_ops = ['x', 'y', 'z']

        component_index = experiment_data.metadata["component_child_index"]

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            scan_indices = child_data.metadata['component_child_index']
            scan_data = [child_data.child_data(idx) for idx in scan_indices]

            if control_state == 0:
                subanalysis = self._analyses[control_state]

                xvar = subanalysis.xvar
                poly_orders = subanalysis.options.poly_orders

                xval = np.array([data.metadata[xvar] for data in scan_data])

            control_basis_components.append([data.analysis_results('hamiltonian_components').value
                                             for data in scan_data])

            if poly_orders is not None:
                for op in target_ops:
                    coeffs = child_data.analysis_results(f'omega_{op}_coeffs').value
                    control_basis_coeffs.append(coeffs)

        # (control, target, xvar)
        control_basis_components = np.transpose(control_basis_components, (0, 2, 1))

        control_eigvals = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) # [c, I/z/ζ]
        control_to_op = np.linalg.inv(control_eigvals)

        hamiltonian_components = np.tensordot(control_to_op, control_basis_components, (1, 0)) * 2

        analysis_results.append(
            AnalysisResultData(
                name='hamiltonian_components_scan',
                value=hamiltonian_components
            )
        )

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            ylabel = 'Hamiltonian component'

            if self.options.plot_angle:
                plotter.set_options(subplots=(2, 1))
                series_params = {f'{cop}{top}': {'canvas': 0}
                                 for cop in control_ops for top in target_ops}
                series_params.update({f'{cop}_angle': {'canvas': 1} for cop in control_ops})
                plotter.set_figure_options(series_params=series_params)
                ylabel = [ylabel, 'arctan(y/x)']

            plotter.set_figure_options(
                xlabel=xvar,
                ylabel=ylabel
            )

            for ic, control_op in enumerate(control_ops):
                for ib, target_op in enumerate(target_ops):
                    plotter.set_series_data(
                        f'{control_op}{target_op}',
                        x_formatted=xval,
                        y_formatted=unp.nominal_values(hamiltonian_components[ic, ib]),
                        y_formatted_err=unp.std_devs(hamiltonian_components[ic, ib])
                    )

                if self.options.plot_angle:
                    plotter.set_series_data(
                        f'{control_op}_angle',
                        x_formatted=xval,
                        y_formatted=np.arctan2(unp.nominal_values(hamiltonian_components[ic, 1]),
                                               unp.nominal_values(hamiltonian_components[ic, 0]))
                    )

            interp_x = np.linspace(xval[0], xval[-1], 100)

        curves = {}

        if self.options.poly_orders is not None:
            for ic, control_op in enumerate(control_ops):
                for ib, target_op in enumerate(target_ops):
                    try:
                        order = PolynomialOrder(self.options.poly_orders[(ic, ib)])
                    except KeyError:
                        continue

                    yval = unp.nominal_values(hamiltonian_components[ic, ib])
                    yval_max = np.amax(np.abs(yval))
                    norm_yval = yval / yval_max

                    fitfunc = sparse_poly_fitfunc(order.powers)
                    p0 = np.zeros_like(order.powers)
                    popt, pcov = sciopt.curve_fit(fitfunc, xval, norm_yval, p0=p0)

                    coeffs = np.full(order.order + 1, ufloat(0., 0.))
                    try:
                        coeffs[order.powers] = correlated_values(nom_values=popt,
                                                                 covariance_mat=pcov)
                    except np.linalg.LinAlgError:
                        coeffs[order.powers] = [ufloat(v, 0.) for v in popt]

                    coeffs *= yval_max

                    name = f'omega_{control_op}{target_op}_coeffs'
                    analysis_results.append(AnalysisResultData(name=name, value=coeffs))

                    if self.options.plot:
                        interp_y = fitfunc(interp_x, *unp.nominal_values(coeffs[order.powers]))
                        plotter.set_series_data(
                            f'{control_op}{target_op}',
                            x_interp=interp_x,
                            y_interp=interp_y
                        )

                        curves[f'{control_op}{target_op}'] = interp_y

        elif poly_orders is not None:
            max_orderplus1 = max(coeffs.shape[0] for coeffs in control_basis_coeffs)
            for idx, coeffs in enumerate(control_basis_coeffs):
                if coeffs.shape[0] < max_orderplus1:
                    pad = max_orderplus1 - coeffs.shape[0]
                    control_basis_coeffs[idx] = np.concatenate(
                        (coeffs, np.full(pad, ufloat(0., 0.)))
                    )

            control_basis_coeffs = np.reshape(control_basis_coeffs, (3, 3, -1))

            hamiltonian_coeffs = np.tensordot(control_to_op, control_basis_coeffs, (1, 0))

            for ic, control_op in enumerate(control_ops):
                for ib, target_op in enumerate(target_ops):
                    coeffs = hamiltonian_coeffs[ic, ib]
                    name = f'omega_{control_op}{target_op}_coeffs'
                    analysis_results.append(AnalysisResultData(name=name, value=coeffs))

                    if self.options.plot:
                        interp_y = poly.polynomial.polyval(interp_x, unp.nominal_values(coeffs))
                        plotter.set_series_data(
                            f'{control_op}{target_op}',
                            x_interp=interp_x,
                            y_interp=interp_y
                        )

                        curves[f'{control_op}{target_op}'] = interp_y

        if self.options.plot:
            if self.options.plot_angle:
                for cop in control_ops:
                    if f'{cop}x' in curves and f'{cop}y' in curves:
                        plotter.set_series_data(
                            f'{cop}_angle',
                            x_interp=interp_x,
                            y_interp=np.arctan2(curves[f'{cop}y'], curves[f'{cop}x'])
                        )

            return analysis_results, [plotter.figure()]
        return analysis_results, []
