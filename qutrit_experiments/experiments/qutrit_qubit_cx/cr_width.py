from collections.abc import Sequence
import logging
from threading import Lock
from typing import Any, Optional, Union
from matplotlib.figure import Figure
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, BackendTiming, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

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
        # (The two schedules are separately added to the circuit and I'm not 100% sure if sharing
        # the parameters would be safe)
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
        options.simul_fit = True
        options.tol = 1.e-4
        options.intercept_min = -np.pi / 2.
        options.intercept_max_wind = 0
        return options

    _lock = Lock()
    _fit_functions_cache = {}

    def _get_p0s(self, unitary_params, control_state):
        unitary_params_n = unp.nominal_values(unitary_params)
        axes = unitary_params_n[:, control_state].copy()
        axes /= np.sqrt(np.sum(np.square(axes), axis=-1))[:, None]
        # Align along a single orientation and take the mean
        mean_ax = np.mean(axes * np.where(axes @ axes[0] < 0., -1., 1.)[:, None], axis=0)
        mean_ax /= np.sqrt(np.sum(np.square(mean_ax)))
        psi = np.arccos(min(max(mean_ax[2], -1.), 1.))
        phi = np.arctan2(mean_ax[1], mean_ax[0])

        # Make a mesh of slope and intercept values as initial value candidates
        slopes = np.linspace(0.01, np.pi, 4)
        intercepts = np.linspace(0., twopi, 4, endpoint=False)
        islope, iint = np.ogrid[:len(slopes), :len(intercepts)]
        p0s = np.empty((2, len(slopes), len(intercepts), 4))
        p0s[..., 0] = slopes[islope]
        p0s[..., 1] = intercepts[iint]
        # Use both orientations
        p0s[0, ..., 2:] = [psi, phi]
        p0s[1, ..., 2:] = [np.pi - psi, np.pi + phi]
        return p0s.reshape(-1, 4)

    def _postprocess_params(self, upopt: np.ndarray, norm: float):
        if upopt[0].n < 0.:
            # Slope must be positive - invert the sign and the axis orientation
            upopt[0:2] *= -1.
            upopt[2] = np.pi - upopt[2]
            upopt[3] += np.pi
        # Keep the intercept within the maximum winding number
        # Intercept must be positive in principle but we allow some slack
        if upopt[1].n < self.options.intercept_min or upopt[1].n > 0.:
            upopt[1] %= twopi * (self.options.intercept_max_wind + 1)
        # Keep phi in [-pi, pi]
        upopt[3] = (upopt[3] + np.pi) % twopi - np.pi
        upopt[0] /= norm

    @classmethod
    def unitary_params(cls, fit_params, wval, npmod=np):
        if npmod is np:
            wval = np.asarray(wval)
        slope, intercept, psi, phi = fit_params
        angle = wval * slope + intercept
        axis = npmod.array([
            npmod.sin(psi) * npmod.cos(phi),
            npmod.sin(psi) * npmod.sin(phi),
            npmod.cos(psi)
        ])
        wval_dims = tuple(range(len(wval.shape)))
        return npmod.expand_dims(angle, axis=-1) * npmod.expand_dims(axis, axis=wval_dims)


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
        fit_params = next(res.value for res in analysis_results if res.name == 'simul_fit_params')
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
        cal_parameter_name: list[str] = ['width', 'rcr_type', 'qutrit_qubit_cx_sign'],
        schedule_name: str = ['cr', None, None],
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
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            width_param_name='width',
            widths=widths
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
        slope, intercept, psi, phi = np.stack([unp.nominal_values(fit_params[ic]) for ic in range(3)],
                                              axis=1)

        phi_offset = phi[[2, 0]] # RCRType [X, X12]
        n_x = np.cos(phi[None, :] - phi_offset[:, None])

        def crcr_x_components(rval):
            # Shape [rcr_type, control]
            v_x = (rval * np.sin(psi))[None, :] * n_x
            # X component in block 0 of CRCR for two RCR types
            crcr_v_0x = 2. * v_x[[0, 1], [2, 0]]
            # X component in block 1 of CRCR for two RCR types
            crcr_v_1x = 2. * np.sum(v_x * np.array([[1., 1., -1.], [-1., 1., 1.]]),
                                    axis=1)
            return crcr_v_0x, crcr_v_1x

        crcr_omega_0, crcr_omega_1 = crcr_x_components(slope)
        crcr_offset_0, crcr_offset_1 = crcr_x_components(intercept)
        crcr_rel_freqs = crcr_omega_1 - crcr_omega_0
        crcr_rel_offsets = crcr_offset_1 - crcr_offset_0
        # rel_freq * width + rel_offset = π + 2nπ
        # -> width = [(π - rel_offset) / rel_freq] % (2π / |rel_freq|)
        widths = ((np.pi - crcr_rel_offsets) / crcr_rel_freqs) % (twopi / np.abs(crcr_rel_freqs))
        # RCR type with shorter width will be used
        rcr_type_index = np.argmin(widths)
        rcr_type = [RCRType.X, RCRType.X12][rcr_type_index]

        cr_width = BackendTiming(self.backend).round_pulse(samples=widths[rcr_type_index])
        # We start with a high CR amp -> width estimate should be on the longer side to allow
        # downward adjustment of amp
        cr_width += self._backend_data.granularity

        cx_sign = np.sign((crcr_rel_freqs * widths + crcr_rel_offsets)[rcr_type_index])

        values = [cr_width, int(rcr_type), cx_sign]
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
