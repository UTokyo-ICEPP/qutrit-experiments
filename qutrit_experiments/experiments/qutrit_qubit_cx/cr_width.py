from collections.abc import Sequence
import logging
from threading import Lock
from typing import Any, Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import BackendTiming, ExperimentData, Options

from ...gates import CrossResonanceGate
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class CRWidthAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for CRWidth.

    Simultaneous fit model is
    [x, y, z] = (slope * w + intercept) * [sin(psi) * cos(phi), sin(psi) * sin(phi), cos(psi)]
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.simul_fit = True
        options.tol = 1.e-4
        options.intercept_min = -np.pi / 2.
        options.intercept_max_wind = 0
        return options

    @classmethod
    def unitary_params(cls, fit_params: np.ndarray, wval: np.ndarray, npmod=np):
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

    def _get_p0s(self, norm_xvals: np.ndarray, xvals_norm: float, unitary_params: np.ndarray):
        axes = unp.nominal_values(unitary_params)
        axes /= np.sqrt(np.sum(np.square(axes), axis=-1))[:, None]
        # Align along a single orientation and take the mean
        mean_ax = np.mean(axes * np.where(axes @ axes[0] < 0., -1., 1.)[:, None], axis=0)
        mean_ax /= np.sqrt(np.sum(np.square(mean_ax)))
        psi = np.arccos(min(max(mean_ax[2], -1.), 1.))
        phi = np.arctan2(mean_ax[1], mean_ax[0])

        # Slope = Rabi frequency. Guess from 100kHz to 5MHz with dt=0.2ns
        slopes_phys = np.array([0.1, 0.5, 1., 5.]) * 1.e+6 * twopi * 2.e-10
        slopes = slopes_phys * xvals_norm
        # Effective total width of the GaussianSquare flanks is ~120
        intercepts = slopes * 120 / xvals_norm
        p0s = np.empty((2, len(slopes), 4))
        p0s[..., 0] = slopes[None, :]
        p0s[..., 1] = intercepts[None, :]
        # Use both orientations
        p0s[0, :, 2:] = [psi, phi]
        p0s[1, :, 2:] = [np.pi - psi, np.pi + phi]
        return p0s.reshape(-1, 4)

    def _postprocess_params(self, upopt: np.ndarray, norm: float):
        logger.debug('Postprocessing fit result %s (norm=%f)', upopt, norm)
        # Slope must be positive - invert the sign and the axis orientation if negative
        if upopt[0].n < 0.:
            logger.debug('Adjusting the slope %s', upopt[0])
            upopt[0:2] *= -1.
            upopt[2] = np.pi - upopt[2]
            upopt[3] += np.pi
        # Keep the intercept within the maximum winding number
        # Intercept must be positive in principle but we allow some slack
        if upopt[1].n < self.options.intercept_min or upopt[1].n > 0.:
            logger.debug('Adjusting the intercept %s', upopt[1])
            upopt[1] %= twopi * (self.options.intercept_max_wind + 1)
        # Keep phi in [-pi, pi]
        upopt[3] = (upopt[3] + np.pi) % twopi - np.pi
        upopt[0] /= norm


class CRRoughWidthCal(BaseCalibrationExperiment, QutritQubitTomographyScan):

    """Rough CR width calibration based on pure-X approximationof the CR unitaries.

    Type X (2):    RCR angles = [θ1+θ0, θ0+θ1, 2θ2]
                  CRCR angles = [2θ2, 2θ0+2θ1-2θ2, 2θ2]
    Type X12 (0):  RCR angles = [2θ0, θ2+θ1, θ1+θ2]
                  CRCR angles = [2θ0, 2θ1+2θ2-2θ0, 2θ0]
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(128., 320., 4)]
        options.require_2x_duration = True
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'rcr_type', 'cx_sign', 'cr_amp'],
        schedule_name: str = ['cr', None, 'cx_geometric_phase', 'cr'],
        auto_update: bool = True,
        widths: Optional[Sequence[float]] = None
    ):
        if widths is None:
            widths = self._default_experiment_options().parameter_values[0]

        super().__init__(
            calibrations,
            physical_qubits,
            CrossResonanceGate(params=[Parameter('width')]),
            'width',
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            values=widths,
            analysis_cls=CRWidthAnalysis
        )
        self.extra_metadata = {}

        self._schedules = [
            calibrations.get_schedule(schedule_name[0], physical_qubits,
                                      assign_params={cal_parameter_name[0]: wval, 'margin': 0})
            for wval in widths
        ]

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)
        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        iwidth = circuit.metadata['composite_index'][0]
        circuit.add_calibration(CrossResonanceGate.gate_name, self.physical_qubits,
                                self._schedules[iwidth],
                                params=[self.experiment_options.parameter_values[0][iwidth]])

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
        rcr_type = [2, 0][rcr_type_index]

        cr_width = BackendTiming(self.backend).round_pulse(samples=widths[rcr_type_index])
        # We start with a high CR amp -> width estimate should be on the longer side to allow
        # downward adjustment of amp
        cr_width += self._backend_data.granularity
        if (req := self.experiment_options.require_2x_duration) or req == 0:
            # We will later need the CR pulse to be longer than two X pulses of the c1 qubit
            if isinstance(req, int):
                qubit = req
            else:
                qubit = self.physical_qubits[0]
            min_width = (2 * self._backend.target['x'][(qubit,)].calibration.duration
                         - self._cals.get_schedule(self._sched_name[0], self.physical_qubits,
                                                   assign_params={self._param_name[0]: 0}).duration)
            while cr_width < min_width:
                cr_width += self._backend_data.granularity

        cx_sign = np.sign(crcr_rel_freqs[rcr_type_index] * cr_width
                          + crcr_rel_offsets[rcr_type_index])

        cr_amp = self._cals.get_parameter_value(self._param_name[3], self.physical_qubits,
                                                     schedule=self._sched_name[3])
        # Linear approximation
        cr_amp /= (cr_width / widths[rcr_type_index])

        values = [cr_width, int(rcr_type), cx_sign, cr_amp]
        for pname, sname, value in zip(self._param_name, self._sched_name, values):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=sname,
                group=self.experiment_options.group
            )
