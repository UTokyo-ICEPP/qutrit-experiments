"""Rough f12 calibration based on spectroscopy."""

from collections.abc import Iterable, Sequence
from typing import Optional, Union
import numpy as np
import lmfit

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasReturnType
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.library import EFSpectroscopy

from ..experiment_mixins import MapToPhysicalQubitsCommonCircuit
from ..framework.calibration_updaters import EFFrequencyUpdater


class EFRoughFrequency(MapToPhysicalQubitsCommonCircuit, EFSpectroscopy):
    """EFSpectroscopy with some tweaks."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse."""
        options = super()._default_experiment_options()
        options.amp = 0.06
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        options = super()._default_run_options()
        options.shots = 2000
        # Something wrong started happening with AVG at some point - returned memory values were
        # e+180 or e-102 or whatever, all over the place. Getting single points & taking the average
        # locally works fine
        options.meas_return = MeasReturnType.SINGLE
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Iterable[float],
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, frequencies, backend=backend, absolute=True)
        self.analysis = GaussianResonanceAnalysis()

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]: # pylint: disable=unused-argument
        center = self._frequencies[len(self._frequencies) // 2]
        freq = center - 8.e+6
        mean = curve.fit_function.sqrt_lorentzian(self._frequencies, amp=1.25, kappa=1.e+6,
                                                  x0=freq, baseline=0.68)
        num_qubits = 1

        memory = []
        for val in mean:
            memory.append(np.tile([[val * np.cos(1.2), val * np.sin(1.2)]], (num_qubits, 1)))
        return memory


class GaussianResonanceAnalysis(curve.CurveAnalysis):
    r"""A class to analyze a resonance peak with a Gaussian function."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        super().__init__(
            models=[
                lmfit.models.ExpressionModel(
                    expr="a * exp(-(x - freq)**2 / (2. * sigma**2)) + b",
                    name="gaussian",
                )
            ],
            name=name,
        )

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plotter.set_figure_options(
            xlabel="Frequency",
            ylabel="Signal (arb. units)",
            xval_unit="Hz",
        )
        options.result_parameters = [curve.ParameterRepr("freq", "f12", "Hz")]
        options.normalization = True
        return options

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            list of fit options that are passed to the fitter function.
        """
        max_abs_y, _ = curve.guess.max_height(curve_data.y, absolute=True)

        user_opt.bounds.set_if_empty(
            a=(-2 * max_abs_y, 2 * max_abs_y),
            sigma=(0, np.ptp(curve_data.x)),
            freq=(min(curve_data.x), max(curve_data.x)),
            b=(-max_abs_y, max_abs_y),
        )
        user_opt.p0.set_if_empty(b=curve.guess.constant_spectral_offset(curve_data.y))

        y_ = curve_data.y - user_opt.p0["b"]

        _, peak_idx = curve.guess.max_height(y_, absolute=True)
        fwhm = curve.guess.full_width_half_max(curve_data.x, y_, peak_idx)

        user_opt.p0.set_if_empty(
            a=(curve_data.y[peak_idx] - user_opt.p0["b"]),
            freq=curve_data.x[peak_idx],
            sigma=fwhm,
        )

        return user_opt


class EFRoughFrequencyCal(BaseCalibrationExperiment, EFRoughFrequency):
    """Calibration experiment for EFRoughFrequency."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        frequencies: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None,
        auto_update: bool = True
    ):
        # cal_parameter_name f12 is hard-coded here because it's set in EFSpectroscopy.__init__
        cal_parameter_name = 'f12'

        if frequencies is None:
            freq_12_est = calibrations.get_parameter_value(cal_parameter_name, physical_qubits[0])
            frequencies = np.linspace(freq_12_est - 20.e+6, freq_12_est + 20.e+6, 41)

        super().__init__(
            calibrations,
            physical_qubits,
            frequencies,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend
        )

        self._updater = EFFrequencyUpdater

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass
