from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
import scipy.optimize as sciopt
from uncertainties import correlated_values, ufloat, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...util.bloch import su2_cartesian, su2_cartesian_axnorm, su2_cartesian_params
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, get_margin, make_crcr_circuit

twopi = 2. * np.pi


class CycledRepeatedCRWidth(QutritQubitTomographyScan):
    """Experiment to simultaneously scan the CR width with control states 0 and 1."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        widths = np.linspace(0., 128., 5)
        margins = np.zeros(5)
        options.parameter_values = [widths, margins, widths, margins]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: RCRType,
        width_param_name: str = 'width',
        margin_param_name: str = 'margin',
        widths: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if widths is None:
            widths = self._default_experiment_options().parameter_values[0]

        assignment = {
            cr_schedules[0].get_parameters(width_param_name)[0]: 0.,
            cr_schedules[0].get_parameters(margin_param_name)[0]: 0.
        }
        risefall_duration = cr_schedules[0].assign_parameters(assignment, inplace=False).duration
        margins = get_margin(risefall_duration, widths, backend)

        # Rename the CR parameters to distinguish crp and crm
        for idx, (prefix, sched) in enumerate(zip(['crp', 'crm'], cr_schedules)):
            assign_params = {sched.get_parameters(pname)[0]: Parameter(f'{prefix}_{pname}')
                             for pname in [width_param_name, margin_param_name]}
            cr_schedules[idx] = sched.assign_parameters(assign_params, inplace=False)

        param_names = [f'{prefix}_{pname}'
                       for prefix in ['crp', 'crm']
                       for pname in [width_param_name, margin_param_name]]
        param_values = [widths, margins, widths, margins]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, None, rcr_type),
                         param_names, param_values, measure_preparations=measure_preparations,
                         control_states=(0, 1), backend=backend)
        self.analysis = CycledRepeatedCRWidthAnalysis([exp.analysis for exp in self._experiments])


class CycledRepeatedCRWidthAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for CycledRepeatedCRWidth."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Fit exp[-i/2 * (slope * width + intercept) * axis.pauli] to the observed unitaries."""
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        # Shape [num_widths, num_control_states(=2), 3]
        unitary_params = next(res.value for res in analysis_results
                              if res.name == 'unitary_parameters')
        unitary_params_n = unp.nominal_values(unitary_params)
        scan_idx = experiment_data.metadata['scan_parameters'].index('crp_width')
        widths = np.array(experiment_data.metadata['scan_values'][scan_idx])
        x_interp = np.linspace(widths[0], widths[-1], 100)

        # Fit in the unitary space
        def axis_from_params(params, npmod=np):
            psi, phi = params[2:]
            return npmod.array([
                npmod.sin(psi) * npmod.cos(phi),
                npmod.sin(psi) * npmod.sin(phi),
                npmod.cos(psi)
            ])

        def angle_from_params(params, widths):
            slope, intercept = params[:2]
            return slope * widths + intercept

        def fidelities(params, ic, npmod=np):
            test_u = su2_cartesian_axnorm(
                axis_from_params(params, npmod=npmod),
                angle_from_params(params, widths),
                npmod=npmod
            )
            # Has to be a jnp array because ic is
            udag = su2_cartesian(npmod.array(unitary_params_n)[:, ic], npmod=npmod).conjugate()
            return npmod.square(npmod.abs(npmod.einsum('wij,wij->w', udag, test_u) / 2.))

        @jax.jit
        def objective(params, ic):
            return 1. - jnp.mean(fidelities(params, ic, npmod=jnp))

        # Negative log fidelity has too many local minima and is not a good objective for fitting,
        # but the uncertainties should be evaluated through this quantity
        # (There must be some mistake in my reasoning though; the uncertainties appear too large)
        def negative_log_fidelity(params, ic):
            return -jnp.sum(jnp.log(fidelities(params, ic, npmod=jnp)))

        solver = jaxopt.GradientDescent(objective, maxiter=10000)

        popt_ufloats = []

        for ic in range(2):
            axes = unitary_params_n[:, ic].copy()
            axes /= np.sqrt(np.sum(np.square(axes), axis=-1))[:, None]

            # Align along a single orientation and take the mean
            mean_ax = np.mean(axes * np.where(axes @ axes[0] < 0., -1., 1.)[:, None], axis=0)
            mean_ax /= np.sqrt(np.sum(np.square(mean_ax)))
            psi = np.arccos(min(max(mean_ax[2], -1.), 1.))
            phi = np.arctan2(mean_ax[1], mean_ax[0])

            # Make a mesh of slope and intercept values as initial value candidates
            slopes = np.linspace(0., twopi / (widths[-1] - widths[0]), 4)
            intercepts = np.linspace(0., twopi, 4, endpoint=False)
            islope, iint = np.ogrid[:len(slopes), :len(intercepts)]
            p0s = np.empty((2, len(slopes), len(intercepts), 4))
            p0s[..., 0] = slopes[islope]
            p0s[..., 1] = intercepts[iint]
            # Use both orientations
            p0s[0, ..., 2:] = [psi, phi]
            p0s[1, ..., 2:] = [np.pi - psi, np.pi + phi]

            fit_result = jax.vmap(solver.run, in_axes=[0, None])(p0s.reshape(-1, 4), ic)
            fvals = jax.vmap(objective, in_axes=[0, None])(fit_result.params, ic)
            iopt = np.argmin(fvals)
            popt = np.array(fit_result.params[iopt])
            if popt[0] < 0.:
                # Slope must be positive - invert the sign and the axis orientation
                popt[0:2] *= -1.
                popt[2] = np.pi - popt[2]
                popt[3] += np.pi
            # Intercept must be in [0, 2pi]
            popt[1] %= twopi
            # Keep phi in [-pi, pi]
            popt[3] = (popt[3] + np.pi) % twopi - np.pi

            pcov = np.linalg.inv(jax.hessian(negative_log_fidelity)(popt, ic))
            popt_ufloats.append(correlated_values(nom_values=popt, covariance_mat=pcov,
                                tags=['slope', 'intercept', 'psi', 'phi']))

            if self.options.plot:
                plotter = CurvePlotter(MplDrawer())
                plotter.set_figure_options(
                    xlabel='CR width',
                    ylabel='angle',
                    ylim=(-np.pi - 0.1, np.pi + 0.1)
                )
                axis = axis_from_params(popt)
                # Constrain theta to [-pi, pi] because the observed unitary parameters are in that
                # domain
                theta = (angle_from_params(popt, x_interp) + np.pi) % twopi - np.pi
                xyz_pred = su2_cartesian_params(su2_cartesian_axnorm(axis, theta))
                for iax, ax in enumerate(['x', 'y', 'z']):
                    plotter.set_series_data(
                        ax,
                        x_formatted=widths,
                        y_formatted=unitary_params_n[:, ic, iax],
                        y_formatted_err=unp.std_devs(unitary_params[:, ic, iax]),
                        x_interp=x_interp,
                        y_interp=xyz_pred[:, iax]
                    )
                figures.append(plotter.figure())

        popt_ufloats = np.array(popt_ufloats)
        analysis_results.append(
            AnalysisResultData('unitary_line_fit_params', value=popt_ufloats)
        )

        # Slope and intercept (with uncertainties) of x[1] - x[0]
        lineparam = popt_ufloats[:, :2]
        psi, phi = popt_ufloats.T[2:]
        slope, intercept = (lineparam[1] * unp.sin(psi[1]) * unp.cos(phi[1])
                            - lineparam[0] * unp.sin(psi[0]) * unp.cos(phi[0]))
        wmin = (np.pi - intercept) / slope
        while wmin.n < 0.:
            wmin += twopi / np.abs(slope.n)
        while (wtest := wmin - twopi / np.abs(slope.n)).n > 0.:
            wmin = wtest
        cx_sign = np.sign(slope.n * wmin.n + intercept.n)

        analysis_results.extend([
            AnalysisResultData(name='cr_width', value=wmin),
            AnalysisResultData(name='cx_sign', value=cx_sign)
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='CR width',
                ylabel=r'$\theta_1 - \theta_0$'
            )
            plotter.set_series_data(
                'angle_diff',
                x_interp=x_interp,
                y_interp=slope.n * x_interp + intercept.n
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class CycledRepeatedCRWidthCal(BaseCalibrationExperiment, CycledRepeatedCRWidth):
    """Calibration experiment for CR width."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin'],
        schedule_name: str = 'cr',
        widths: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name}
        cr_schedules = [calibrations.get_schedule(schedule_name, physical_qubits,
                                                  assign_params=assign_params)]
        for pname in ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']:
            # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
            assign_params[pname] = np.pi
        cr_schedules.append(calibrations.get_schedule(schedule_name, physical_qubits,
                                                      assign_params=assign_params))

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            width_param_name=cal_parameter_name[0],
            margin_param_name=cal_parameter_name[1],
            widths=widths,
            measure_preparations=measure_preparations
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        width = experiment_data.analysis_results('cr_width', block=False).value.n
        null_sched = self._cals.get_schedule(self._sched_name, self.physical_qubits,
                                             assign_params={p: 0. for p in self._param_name})
        margin = get_margin(null_sched.duration, width, self._backend)

        for pname, value in zip(self._param_name, [width, margin]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )
