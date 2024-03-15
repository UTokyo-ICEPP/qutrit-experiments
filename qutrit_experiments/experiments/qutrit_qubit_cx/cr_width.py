from collections.abc import Sequence
import logging
from threading import Lock
from typing import Any, Optional, Union
from matplotlib.figure import Figure
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework import AnalysisResultData, BackendTiming, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...util.bloch import so3_cartesian, so3_cartesian_axnorm, so3_cartesian_params
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, get_cr_schedules, get_margin, make_cr_circuit, make_crcr_circuit

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class CRRoughWidth(QutritQubitTomographyScan):
    """Qutrit-qubit UT scan to make a rough estimate of what the CR width for CRCR will be."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(128., 320., 4)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        width_param_name: str = 'width',
        widths: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if widths is None:
            widths = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits, make_cr_circuit(physical_qubits, schedule),
                         width_param_name, widths, measure_preparations=measure_preparations,
                         backend=backend)
        self.analysis = CRWidthAnalysis([exp.analysis for exp in self._experiments])
        self.extra_metadata = {}

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)
        return metadata


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
        reparametrized = []
        for prefix, sched in zip(['crp', 'crm'], cr_schedules):
            assign_params = {sched.get_parameters(pname)[0]: Parameter(f'{prefix}_{pname}')
                             for pname in [width_param_name, margin_param_name]}
            reparametrized.append(sched.assign_parameters(assign_params, inplace=False))
        cr_schedules = tuple(reparametrized)

        param_names = [f'{prefix}_{pname}'
                       for prefix in ['crp', 'crm']
                       for pname in [width_param_name, margin_param_name]]
        param_values = [widths, margins, widths, margins]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, None, rcr_type),
                         param_names, param_values, measure_preparations=measure_preparations,
                         control_states=(0, 1), backend=backend,
                         analysis_cls=CycledRepeatedCRWidthAnalysis)
        self.analysis.set_options(width_name='crp_width')


class CRWidthAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for CRWidth."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.parallelize_on_thread = True
        options.width_name = 'width'
        options.tol = 1.e-4
        options.intercept_min = -np.pi / 2.
        options.intercept_max_wind = 0
        return options

    @staticmethod
    def axis_from_params(params, npmod=np):
        psi, phi = params[2:]
        return npmod.array([
            npmod.sin(psi) * npmod.cos(phi),
            npmod.sin(psi) * npmod.sin(phi),
            npmod.cos(psi)
        ])

    @staticmethod
    def angle_from_params(params, wval):
        slope, intercept = params[:2]
        return wval * slope + intercept

    @classmethod
    def setup_fitter(cls, params, widths, indices, expvals, expvals_err, prep_unitary):
        def objective(params, widths, indices, expvals, expvals_err, prep_unitary):
            axis = cls.axis_from_params(params, npmod=jnp)
            angles = cls.angle_from_params(params, widths)
            unitaries = prep_unitary @ so3_cartesian_axnorm(axis, angles, npmod=jnp)
            r_elements = unitaries[indices]
            return jnp.sum(jnp.square((r_elements - expvals) / expvals_err))

        key = (widths.shape[0], expvals.shape[0])
        args = (widths, indices, expvals, expvals_err, prep_unitary)
        in_axes = [0] + [None] * len(args)
        vobj = jax.jit(jax.vmap(objective, in_axes=in_axes)).lower(params, *args).compile()
        solver = jaxopt.GradientDescent(objective, maxiter=10000, tol=1.e-4)
        vsolve = jax.jit(jax.vmap(solver.run, in_axes=in_axes)).lower(params, *args).compile()
        hess = jax.jit(jax.hessian(objective)).lower(params[0], *args).compile()
        cls._fit_functions[key] = (vobj, vsolve, hess)

    @classmethod
    def fit_functions(cls, params, widths, indices, expvals, expvals_err, prep_unitary):
        key = (widths.shape[0], expvals.shape[0])
        with cls._lock:
            if (fns := cls._fit_functions.get(key)) is None:
                cls.setup_fitter(params, widths, indices, expvals, expvals_err, prep_unitary)
                fns = cls._fit_functions[key]
        return fns

    _lock = Lock()
    _fit_functions = {}

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Fit exp[-i/2 * (slope * width + intercept) * axis.pauli] to the expectation values."""
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        # Shape [num_widths, num_control_states, 3]
        unitary_params = next(res.value for res in analysis_results
                              if res.name == 'unitary_parameters')
        unitary_params_n = unp.nominal_values(unitary_params)
        scan_idx = experiment_data.metadata['scan_parameters'].index(self.options.width_name)
        widths = np.array(experiment_data.metadata['scan_values'][scan_idx])
        # Use normalized widths throughout
        widths_norm = widths[-1] - widths[0]
        norm_widths = widths / widths_norm

        # Use the first child data for control state information
        first = experiment_data.child_data(0)
        control_states = first.metadata['control_states']
        index_state_map = {idx: first.child_data(idx).metadata['control_state']
                           for idx in first.metadata['component_child_index']
                           if not first.child_data(idx).metadata.get('state_preparation', False)}
        try:
            prep_params = first.analysis_results('prep_parameters').value
        except ExperimentEntryNotFound:
            prep_unitaries = {}
        else:
            prep_unitaries = {state: so3_cartesian(unp.nominal_values(params))
                              for state, params in prep_params.items()}

        if (data_processor := self.options.data_processor) is None:
            data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])

        axes = ['x', 'y', 'z']

        expvals = {state: [] for state in control_states}
        iwidths = {state: [] for state in control_states}
        initial_states = {state: [] for state in control_states}
        meas_bases = {state: [] for state in control_states}
        for datum, expval in zip(experiment_data.data(), data_processor(experiment_data.data())):
            scan_metadata = datum['metadata']
            qqt_metadata = scan_metadata['composite_metadata'][0]
            ut_metadata = qqt_metadata['composite_metadata'][0]
            if (control_state := index_state_map.get(qqt_metadata['composite_index'][0])) is None:
                continue

            iwidth = scan_metadata['composite_index'][0]
            expvals[control_state].append(expval)
            iwidths[control_state].append(iwidth)
            initial_states[control_state].append(axes.index(ut_metadata['initial_state']))
            meas_bases[control_state].append(axes.index(ut_metadata['meas_basis']))

        # Fit in the unitary space
        popts = {}
        popt_ufloats = {}
        for control_state in control_states:
            axes = unitary_params_n[:, control_state].copy()
            axes /= np.sqrt(np.sum(np.square(axes), axis=-1))[:, None]
            # Align along a single orientation and take the mean
            mean_ax = np.mean(axes * np.where(axes @ axes[0] < 0., -1., 1.)[:, None], axis=0)
            mean_ax /= np.sqrt(np.sum(np.square(mean_ax)))
            psi = np.arccos(min(max(mean_ax[2], -1.), 1.))
            phi = np.arctan2(mean_ax[1], mean_ax[0])

            # Make a mesh of slope and intercept values as initial value candidates
            slopes = np.linspace(0., np.pi, 4)
            intercepts = np.linspace(0., twopi, 4, endpoint=False)
            islope, iint = np.ogrid[:len(slopes), :len(intercepts)]
            p0s = np.empty((2, len(slopes), len(intercepts), 4))
            p0s[..., 0] = slopes[islope]
            p0s[..., 1] = intercepts[iint]
            # Use both orientations
            p0s[0, ..., 2:] = [psi, phi]
            p0s[1, ..., 2:] = [np.pi - psi, np.pi + phi]
            p0s = p0s.reshape(-1, 4)

            indices = (iwidths[control_state], meas_bases[control_state],
                       initial_states[control_state])
            expvals_n = unp.nominal_values(expvals[control_state])
            expvals_e = unp.std_devs(expvals[control_state])
            fit_args = (norm_widths, indices, expvals_n, expvals_e / np.mean(expvals_e),
                        prep_unitaries.get(control_state, np.eye(3)))

            vobj, vsolve, hess = self.fit_functions(p0s, *fit_args)

            fit_result = vsolve(p0s, *fit_args)
            fvals = vobj(fit_result.params, *fit_args)
            iopt = np.argmin(fvals)
            popt = np.array(fit_result.params[iopt])
            logger.debug('Control state %d optimal parameters %s', control_state, popt)

            if popt[0] < 0.:
                # Slope must be positive - invert the sign and the axis orientation
                popt[0:2] *= -1.
                popt[2] = np.pi - popt[2]
                popt[3] += np.pi
            # Keep the intercept within the maximum winding number
            # Intercept must be positive in principle but we allow some slack
            if popt[1] < self.options.intercept_min or popt[1] > 0.:
                popt[1] %= twopi * (self.options.intercept_max_wind + 1)
            # Keep phi in [-pi, pi]
            popt[3] = (popt[3] + np.pi) % twopi - np.pi
            popts[control_state] = popt

            logger.debug('Control state %d adjusted parameters %s', control_state, popt)

            hess_args = (norm_widths, indices, expvals_n, expvals_e,
                         prep_unitaries.get(control_state, np.eye(3)))
            pcov = np.linalg.inv(hess(popt, *hess_args))
            popt_ufloats[control_state] = np.array(correlated_values(popt, pcov))
            popt_ufloats[control_state][0] /= widths_norm
            popt[0] /= widths_norm

        analysis_results.append(
            AnalysisResultData('unitary_linear_fit_params', value=popt_ufloats)
        )

        if self.options.plot:
            x_interp = np.linspace(widths[0], widths[-1], 100)
            xyz_preds = {
                ic: so3_cartesian_params(
                    so3_cartesian_axnorm(
                        self.axis_from_params(popts[ic]),
                        self.angle_from_params(popts[ic], x_interp)
                    )
                )
                for ic in control_states
            }
            for iax, figure in enumerate(figures[:3]):
                ax = figure.axes[0]
                for ic in control_states:
                    ax.plot(x_interp, xyz_preds[ic][:, iax])

        return analysis_results, figures


class CycledRepeatedCRWidthAnalysis(CRWidthAnalysis):
    """Analysis for CycledRepeatedCRWidth."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names.append('angle_diff')
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        scan_idx = experiment_data.metadata['scan_parameters'].index(self.options.width_name)
        widths = np.array(experiment_data.metadata['scan_values'][scan_idx])
        x_interp = np.linspace(widths[0], widths[-1], 100)
        fit_params = next(res.value for res in analysis_results
                          if res.name == 'unitary_linear_fit_params')
        # Slope and intercept (with uncertainties) of x[1] - x[0]
        slope, intercept, psi, phi = [
            np.array([fit_params[0][ip], fit_params[1][ip]]) for ip in range(4)
        ]
        dxslope = np.diff(slope * unp.sin(psi) * unp.cos(phi))[0]
        dxintercept = np.diff(intercept * unp.sin(psi) * unp.cos(phi))[0]
        wmin = (np.pi - dxintercept) / dxslope
        while wmin.n < 0.:
            wmin += twopi / np.abs(dxslope.n)
        while (wtest := wmin - twopi / np.abs(dxslope.n)).n > 0.:
            wmin = wtest
        cx_sign = np.sign(dxslope.n * wmin.n + dxintercept.n)

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
                y_interp=dxslope.n * x_interp + dxintercept.n
            )
            figures.append(plotter.figure())

        return analysis_results, figures


class CRRoughWidthCal(BaseCalibrationExperiment, CRRoughWidth):
    """Rough CR width calibration based on pure-X approximationof the CR unitaries.

    Type X (2):    RCR angles = [θ1+θ0, θ0+θ1, 2θ2]
                  CRCR angles = [2θ2, 2θ0+2θ1-2θ2, 2θ2]
    Type X12 (0):  RCR angles = [2θ0, θ2+θ1, θ1+θ2]
                  CRCR angles = [2θ0, 2θ1+2θ2-2θ0, 2θ0]
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'rcr_type'],
        schedule_name: str = ['cr', None],
        auto_update: bool = True,
        widths: Optional[Sequence[float]] = None
    ):
        width = Parameter('width')
        assign_params = {cal_parameter_name[0]: width, 'margin': 0}
        schedule = calibrations.get_schedule(schedule_name[0], physical_qubits,
                                             assign_params=assign_params)
        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            'width',
            widths,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        fit_params = experiment_data.analysis_results('unitary_linear_fit_params',
                                                      block=False).value
        slope, intercept, psi, phi = np.stack([unp.nominal_values(fit_params[ic]) for ic in range(3)],
                                              axis=1)

        # Approximate transverse Hamiltonian and offset
        # Shape [control, xy]
        h_t = (slope * np.sin(psi))[:, None] * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        offset_t = (intercept * np.sin(psi))[:, None] * np.stack([np.cos(phi), np.sin(phi)], axis=1)
        # Rotate to align RCR non-participating state to +X
        phis = phi[[2, 0]] # RCRType [X, X12]
        rotation = np.array([[np.cos(phis), np.sin(phis)],
                             [-np.sin(phis), np.cos(phis)]]).transpose((2, 0, 1))
        # Rotated X Hamiltonian
        # Shape [rcr_type, control]
        h_x = np.einsum('rij,cj->rci', rotation, h_t)[..., 0]
        offset_x = np.einsum('rij,cj->rci', rotation, offset_t)[..., 0]
        # Angle per CR width of block 0 of CRCR for two RCR types
        crcr_omega_0 = 2. * h_x[[0, 1], [2, 0]]
        # Angle per CR width of block 1 of CRCR for two RCR types
        crcr_omega_1 = 2. * np.sum(h_x * np.array([[1., 1., -1.], [-1., 1., 1.]]),
                                    axis=1)
        crcr_rel_freqs = crcr_omega_1 - crcr_omega_0
        crcr_offset_0 = 2. * offset_x[[0, 1], [2, 0]]
        crcr_offset_1 = 2. * np.sum(offset_x * np.array([[1., 1., -1.], [-1., 1., 1.]]),
                                    axis=1)
        crcr_rel_offsets = crcr_offset_1 - crcr_offset_0
        # Whichever RCR type with larger angular rate will be used
        rcr_type_index = np.argmax(np.abs(crcr_rel_freqs))
        rcr_type = [RCRType.X, RCRType.X12][rcr_type_index]
        # Compute the width accounting for the gaussiansquare flanks
        rel_offset = crcr_rel_offsets[rcr_type_index]
        rel_freq = crcr_rel_freqs[rcr_type_index]
        samples = ((np.pi - rel_offset) / rel_freq) % (twopi / np.abs(rel_freq))

        cr_width = BackendTiming(self.backend).round_pulse(samples=samples)
        # We start with a high CR amp -> width estimate should be on the longer side to allow
        # downward adjustment of amp
        cr_width += self._backend_data.granularity

        values = [cr_width, int(rcr_type)]
        for pname, sname, value in zip(self._param_name, self._sched_name, values):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=sname,
                group=self.experiment_options.group
            )


class CycledRepeatedCRWidthCal(BaseCalibrationExperiment, CycledRepeatedCRWidth):
    """Calibration experiment for CR width."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin', 'qutrit_qubit_cx_sign'],
        schedule_name: list[Union[str, None]] = ['cr', 'cr', None],
        widths: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        cr_schedules = get_cr_schedules(calibrations, physical_qubits,
                                        free_parameters=cal_parameter_name[:2])
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
        null_sched = self._cals.get_schedule(self._sched_name[0], self.physical_qubits,
                                             assign_params={p: 0. for p in self._param_name[:2]})
        margin = get_margin(null_sched.duration, width, self._backend)
        cx_sign = experiment_data.analysis_results('cx_sign', block=False).value

        for pname, sname, value in zip(self._param_name, self._sched_name,
                                       [width, margin, cx_sign]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=sname,
                group=self.experiment_options.group
            )
