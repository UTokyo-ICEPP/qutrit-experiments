from collections.abc import Iterable, Sequence
from typing import Optional, Union
import numpy as np
import lmfit

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasReturnType
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import ExperimentData, Options, AnalysisResultData, BackendTiming
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import Frequency
from qiskit_experiments.library import EFSpectroscopy
import qiskit_experiments.curve_analysis as curve

from ..transpilation import replace_calibration_and_metadata


class EFRoughFrequency(EFSpectroscopy):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default option values used for the spectroscopy pulse.
        Experiment Options:
            amp (float): The amplitude of the spectroscopy pulse. Defaults to 0.1 and must
                be between 0 and 1.
            duration (int): The duration of the spectroscopy pulse. Defaults to 1024 samples.
            sigma (float): The standard deviation of the flanks of the spectroscopy pulse.
                Defaults to 256.
            width (int): The width of the flat-top part of the GaussianSquare pulse.
                Defaults to 0.
        """
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
        """See :class:`QubitSpectroscopy` for detailed documentation.
        Args:
            qubit: The qubit on which to run spectroscopy.
            frequencies: The frequencies to scan in the experiment, in Hz.
            backend: Optional, the backend to run the experiment on.
        """
        super().__init__(physical_qubits, frequencies, backend=backend, absolute=True)
        self.analysis = GaussianResonanceAnalysis()
        self.analysis.plotter.set_figure_options(ylim=(-1.3, 1.3))

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                self._backend_data.coupling_map)

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        center = self._frequencies[len(self._frequencies) // 2]
        freq = center - 8.e+6
        #freq2 = center + 9.e+6
        #mean = double_resonance(self._frequencies,
        #                        1.25, 1.e+6, freq, 0.68, 2.0, 1.2e+6, freq2)
        mean = curve.fit_function.sqrt_lorentzian(self._frequencies, amp=1.25, kappa=1.e+6,
                                                  x0=freq, baseline=0.68)

        num_qubits = 1

        memory = list()

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

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations.

        We are not using update_library.Frequency here (as is done in RoughFrequencyCal) because
        we need to specify the schedule name in order to make the calibration reflected in the
        instruction map (Frequency updater updates with schedule=None, which makes sense because
        the frequency is not supposed to be schedule-specific, but that causes the Calibrations
        class to ignore the parameter).
        """
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        Frequency.update(self._cals, experiment_data, result_index=result_index,
                         parameter=self._param_name, group=group, fit_parameter='f12')
