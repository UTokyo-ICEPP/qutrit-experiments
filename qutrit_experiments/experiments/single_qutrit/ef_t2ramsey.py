"""Ramsey experiment for 1-2 space to see the modulated cosine curve arising from charge
fluctuation."""
from collections.abc import Sequence
from typing import Optional
import lmfit
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit.library import SXGate
from qiskit.providers import Backend
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import Options
from qiskit_experiments.library import T2Ramsey

from ...experiment_mixins.ef_space import EFSpaceExperiment
from ...gates import SX12Gate

twopi = 2. * np.pi


class EFT2Ramsey(EFSpaceExperiment, T2Ramsey):
    """Ramsey experiment for 1-2 space to see the modulated cosine curve arising from charge
    fluctuation."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Sequence[float],
        backend: Backend = None,
        osc_freq: float = 0.0
    ):
        super().__init__(physical_qubits, delays, backend=backend, osc_freq=osc_freq)
        self.analysis = EFT2RamseyAnalysis()


class EFT2RamseyAnalysis(curve.CurveAnalysis):
    """Curve analysis with cosine envelope."""
    @staticmethod
    def modulated_cosine(x, amp, freq, beat_freq, phase, beat_phase, base):
        return (0.5 * amp * unp.cos(twopi * freq * x + phase) # pylint: disable=no-member
                * unp.cos(twopi * beat_freq * x + beat_phase)) + base # pylint: disable=no-member

    @classmethod
    def _default_options(cls) -> Options:
        default_options = super()._default_options()
        default_options.result_parameters.append("beat_freq")
        return default_options

    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[lmfit.Model(self.modulated_cosine, name='modulated_cosine')],
            name=name
        )
        self.set_options(
            bounds={
                'amp': (0., 2.),
                'freq': (0., np.inf),
                'beat_freq': (0., np.inf),
                'phase': (-np.pi, np.pi),
                'beat_phase': (0., np.pi),
                'base': (0., 1.)
            }
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> list[curve.FitOptions]:
        # to run FFT x interval should be identical
        sampling_interval = np.unique(np.round(np.diff(curve_data.x), decimals=20))

        if len(sampling_interval) != 1:
            # resampling with minimum xdata interval
            sampling_interval = np.min(sampling_interval)
            x_ = np.arange(curve_data.x[0], curve_data.x[-1], sampling_interval)
            y_ = np.interp(x_, xp=curve_data.x, fp=curve_data.y)
        else:
            sampling_interval = sampling_interval[0]
            x_ = curve_data.x
            y_ = curve_data.y

        fft_data = np.fft.fft(y_ - np.average(y_))
        freqs = np.fft.fftfreq(len(x_), sampling_interval)

        positive_freqs = freqs[freqs >= 0]
        positive_fft_data = fft_data[freqs >= 0]

        freqs_sorted = positive_freqs[np.argsort(np.abs(positive_fft_data))]

        base = np.mean(curve_data.y)

        user_opt.p0.set_if_empty(
            base=base,
            amp=curve.guess.max_height(curve_data.y - base, absolute=True)[0],
            freq=(freqs_sorted[-1] + freqs_sorted[-2]) * 0.5,
            beat_freq=(freqs_sorted[-1] - freqs_sorted[-2]) * 0.5
        )

        options = []
        for phase_guess in np.linspace(-np.pi, np.pi, 9):
            for beat_phase_guess in np.linspace(0., np.pi, 5):
                new_opt = user_opt.copy()
                new_opt.p0.update(phase=phase_guess, beat_phase=beat_phase_guess)
                options.append(new_opt)

        return options
