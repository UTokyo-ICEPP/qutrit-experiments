"""Characterization experiments for the SiZZle effect."""
from collections.abc import Sequence
from typing import Any, Optional, Union
from matplotlib.figure import Figure
import numpy as np
import lmfit
from uncertainties import unumpy as unp
from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.curve_analysis.utils import convert_lmfit_result, eval_with_uncertainties
from qiskit_experiments.framework import AnalysisResultData, BackendData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from ..pulse_library import ModulatedGaussianSquare
from .spectator_ramsey import SpectatorRamseyXY
from .zzramsey import QutritZZRamsey

twopi = 2. * np.pi


def build_sizzle_schedule(
    frequency: float,
    control: int,
    target: int,
    backend: Backend,
    amplitudes: tuple[float, float] = None,
    control_phase_offset: float = 0.,
    control_channel: Optional[pulse.ControlChannel] = None,
    target_channel: Optional[pulse.DriveChannel] = None,
    pre_delay: Optional[int] = None
) -> ScheduleBlock:
    """Build a siZZle schedule using two ModulatedGaussianSquare pulses."""
    if amplitudes is None:
        amplitudes = (0.1, 0.1)

    backend_data = BackendData(backend)
    target_freq = backend_data.drive_freqs[target]
    detuning = (frequency - target_freq) * backend_data.dt

    if control_channel is None:
        control_channel = backend_data.control_channel((control, target))[0]
    if target_channel is None:
        target_channel = backend_data.drive_channel(target)

    delay = Parameter('delay')
    with pulse.build(name='delay', default_alignment='left') as delay_sched:
        if pre_delay:
            pulse.delay(pre_delay, control_channel)
            pulse.delay(pre_delay, target_channel)

        pulse.play(
            ModulatedGaussianSquare(
                duration=(delay + 256),
                amp=amplitudes[0],
                sigma=64,
                freq=detuning,
                width=delay,
                angle=control_phase_offset
            ),
            control_channel
        )
        pulse.play(
            ModulatedGaussianSquare(
                duration=(delay + 256),
                amp=amplitudes[1],
                sigma=64,
                freq=detuning,
                width=delay
            ),
            target_channel
        )

    return delay_sched


class SiZZleRamsey(SpectatorRamseyXY):
    """SpectatorRamseyXY experiment for SiZZle."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        frequency: float,
        amplitudes: Optional[tuple[float, float]] = None,
        control_phase_offset: float = 0.,
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        delay_schedule = build_sizzle_schedule(frequency, physical_qubits[0], physical_qubits[1],
                                               backend, amplitudes=amplitudes,
                                               control_phase_offset=control_phase_offset,
                                               control_channel=channels[0],
                                               target_channel=channels[1])

        super().__init__(physical_qubits, control_state, delays=delays, osc_freq=osc_freq,
                         delay_schedule=delay_schedule, extra_metadata=extra_metadata,
                         backend=backend)
        self.set_experiment_options(reverse_qubit_order=True)


class SiZZle(QutritZZRamsey):
    """QutritZZRamsey experiment for SiZZle."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        amplitudes: Optional[tuple[float, float]] = None,
        control_phase_offset: float = 0.,
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        delay_schedule = build_sizzle_schedule(frequency, physical_qubits[0], physical_qubits[1],
                                               backend, amplitudes=amplitudes,
                                               control_phase_offset=control_phase_offset,
                                               control_channel=channels[0],
                                               target_channel=channels[1])

        super().__init__(physical_qubits, delays=delays, osc_freq=osc_freq,
                         delay_schedule=delay_schedule, extra_metadata=extra_metadata,
                         backend=backend)
        for exp in self.component_experiment():
            exp.set_experiment_options(reverse_qubit_order=True)


class SiZZleFrequencyScan(BatchExperiment):
    """Frequency scan of SiZZle."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Sequence[float],
        amplitudes: Optional[tuple[float, float]] = None,
        control_phase_offset: float = 0.,
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for freq in frequencies:
            exp = SiZZle(physical_qubits, frequency=freq, amplitudes=amplitudes,
                         control_phase_offset=control_phase_offset, channels=channels,
                         delays=delays, osc_freq=osc_freq, extra_metadata={'frequency': freq},
                         backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=SiZZleFrequencyScanAnalysis(analyses))


class SiZZleFrequencyScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZleFrequencyScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list['mpl.figure.Figure']]:
        """Plot Iz, zz, and ζz components as functions of siZZle frequency."""
        if not self.options.plot:
            return analysis_results, figures

        component_index = experiment_data.metadata["component_child_index"]

        plotter = CurvePlotter(MplDrawer())
        plotter.set_figure_options(
            xlabel='siZZle frequency (Hz)',
            ylabel='Hamiltonian components (rad/s)'
        )

        frequencies = np.empty(len(component_index))
        components = np.empty((3, len(component_index)), dtype=object)

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            frequencies[ichild] = child_data.metadata['frequency']
            components[:, ichild] = child_data.analysis_results('omega_zs').value

        for ic, label in enumerate(['Iz', 'zz', 'ζz']):
            plotter.set_series_data(
                label,
                x_formatted=np.array(frequencies),
                y_formatted=unp.nominal_values(components[ic]),
                y_formatted_err=unp.std_devs(components[ic])
            )

        return analysis_results, figures + [plotter.figure()]


class SiZZlePhaseScan(BatchExperiment):
    """SiZZle phase scan experiment."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.phase_offsets = np.linspace(0., twopi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        control_phase_offsets: Optional[Sequence[float]] = None,
        amplitudes: Optional[tuple[float, float]] = None,
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        if (offset_values := control_phase_offsets) is None:
            offset_values = self._default_experiment_options().phase_offsets

        experiments = []
        analyses = []

        for offset in offset_values:
            exp = SiZZle(physical_qubits, frequency=frequency, amplitudes=amplitudes,
                         control_phase_offset=offset, channels=channels, delays=delays,
                         osc_freq=osc_freq, extra_metadata={'control_phase_offset': offset},
                         backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=SiZZlePhaseScanAnalysis(analyses))

        if control_phase_offsets is not None:
            self.set_experiment_options(phase_offsets=control_phase_offsets)


class SiZZlePhaseScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZlePhaseScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.return_fit_parameters = False
        options.expected_signs = [-1, 1, -1] # Iz, zz, ζz
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list['mpl.figure.Figure']]:
        """Plot Iz, zz, and ζz components as functions of siZZle phase offset."""
        component_index = experiment_data.metadata["component_child_index"]

        phase_offsets = np.empty(len(component_index))
        components = np.empty((3, len(component_index)), dtype=object)

        ops = ['Iz', 'zz', 'ζz']

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            phase_offsets[ichild] = child_data.metadata['control_phase_offset']
            components[:, ichild] = child_data.analysis_results('omega_zs').value

        models = list(lmfit.models.ExpressionModel(
            expr=f'amp_{label} * cos(x + cr_phase_offset) + base_{label}'
        ) for label in ops)
        yvals = np.array(list(unp.nominal_values(comp) for comp in components))
        with np.errstate(divide='ignore'):
            weights = np.array(list(1. / unp.std_devs(comp) for comp in components))

        def objective(params):
            ys = []
            for ic, model in enumerate(models):
                ys.append(model._residual(
                    params=params,
                    data=yvals[ic],
                    weights=weights[ic],
                    x=phase_offsets
                ))

            return np.concatenate(ys)

        params = lmfit.Parameters()
        params.add(name='cr_phase_offset', value=0., min=-np.pi, max=np.pi)

        for ic, label in enumerate(ops):
            base_guess = np.mean(yvals[ic])
            params.add(name=f'base_{label}', value=base_guess, min=-np.inf, max=np.inf)

            amp_guess = (np.amax(yvals[ic]) - np.amin(yvals[ic])) / 2.
            amp_guess *= self.options.expected_signs[ic]
            if amp_guess < 0.:
                bounds = (-np.inf, 0.)
            else:
                bounds = (0., np.inf)
            params.add(name=f'amp_{label}', value=amp_guess, min=bounds[0], max=bounds[1])

        result = lmfit.minimize(fcn=objective, params=params, method='least_squares')
        fit_data = convert_lmfit_result(result, models, phase_offsets, yvals)

        if fit_data.success:
            quality = 'good'

            analysis_results.append(AnalysisResultData(
                name='cr_phase_offset',
                value=fit_data.ufloat_params['cr_phase_offset'],
                chisq=fit_data.reduced_chisq,
                quality=quality
            ))
        else:
            quality = 'bad'

        if self.options.return_fit_parameters:
            analysis_results.append(AnalysisResultData(
                name=PARAMS_ENTRY_PREFIX + self.__class__.__name__,
                value=fit_data,
                quality=quality,
            ))

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='siZZle phase offset',
                ylabel='Hamiltonian components (rad/s)'
            )

            x_interp = np.linspace(phase_offsets[0], phase_offsets[-1], 100)

            for ic, label in enumerate(ops):
                plotter.set_series_data(
                    label,
                    x_formatted=phase_offsets,
                    y_formatted=yvals[ic],
                    y_formatted_err=unp.std_devs(components[ic])
                )

                if fit_data.success:
                    y_data_with_uncertainty = eval_with_uncertainties(
                        x=x_interp,
                        model=models[ic],
                        params=fit_data.ufloat_params,
                    )
                    y_interp = unp.nominal_values(y_data_with_uncertainty)

                    plotter.set_series_data(
                        label,
                        x_interp=x_interp,
                        y_interp=y_interp
                    )
                    if fit_data.covar is not None:
                        y_interp_err = unp.std_devs(y_data_with_uncertainty)
                        if np.isfinite(y_interp_err).all():
                            plotter.set_series_data(
                                label,
                                y_interp_err=y_interp_err
                            )

            figures.append(plotter.figure())

        return analysis_results, figures


class SiZZleAmplitudeScan(BatchExperiment):
    """SiZZle amplitude scan experiment."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = (0.1, np.linspace(0.01, 0.06, 6))
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        control_phase_offset: float = 0.,
        amplitudes: Optional[Union[tuple[Sequence[float], float], tuple[float, Sequence[float]]]] = None, # pylint: disable=line-too-long
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        if (amplitude_values := amplitudes) is None:
            amplitude_values = self._default_experiment_options().amplitudes

        try:
            len(amplitude_values[0])
        except TypeError:
            scan_qubit = 1
        else:
            scan_qubit = 0

        experiments = []
        analyses = []

        amp_argument = [None] * 2
        amp_argument[1 - scan_qubit] = amplitude_values[1 - scan_qubit]

        for amplitude in amplitude_values[scan_qubit]:
            amp_argument[scan_qubit] = amplitude
            extra_metadata = {
                'scan_qubit': scan_qubit,
                'amplitudes': list(amp_argument)
            }
            exp = SiZZle(physical_qubits, frequency=frequency, amplitudes=amp_argument,
                         control_phase_offset=control_phase_offset, channels=channels,
                         delays=delays, osc_freq=osc_freq, extra_metadata=extra_metadata,
                         backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=SiZZleAmplitudeScanAnalysis(analyses))

        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)


class SiZZleAmplitudeScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZleAmplitudeScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.return_fit_parameters = False
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list['mpl.figure.Figure']]:
        """Plot Iz, zz, and ζz components as functions of siZZle amplitude."""
        component_index = experiment_data.metadata["component_child_index"]

        amplitudes = np.empty(len(component_index))
        components = np.empty((3, len(component_index)), dtype=object)

        ops = ['Iz', 'zz', 'ζz']

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            scan_qubit = child_data.metadata['scan_qubit']
            amplitudes[ichild] = child_data.metadata['amplitudes'][scan_qubit]
            components[:, ichild] = child_data.analysis_results('omega_zs').value

        fit_results = [None] * 3
        fit_data = [None] * 3
        for ic, op in enumerate(ops):
            yval = unp.nominal_values(components[ic])

            if ic == 0:
                if scan_qubit == 0:
                    model = lmfit.models.ConstantModel()
                    c = np.mean(yval)
                    params = model.make_params(c=c)

                else:
                    model = lmfit.models.QuadraticModel()
                    diff_amp = np.diff(amplitudes)
                    a = np.mean(np.diff(np.diff(yval) / diff_amp) / diff_amp[:-1] / 2.)
                    c = yval[0] - a * np.square(amplitudes[0])
                    params = model.make_params(a=a, b={'value': 0., 'vary': False}, c=c)
            else:
                model = lmfit.models.LinearModel()
                slope = (yval[-1] - yval[0]) / (amplitudes[-1] - amplitudes[0])
                intercept = yval[0] - slope * amplitudes[0]
                params = model.make_params(slope=slope, intercept=intercept)

            fit_results[ic] = model.fit(yval, params, x=amplitudes)
            fit_data[ic] = convert_lmfit_result(fit_results[ic], [model], amplitudes, yval)

            if fit_data[ic].success:
                quality = 'good'

                analysis_results.append(AnalysisResultData(
                    name=f'ω_{op}_coeffs',
                    value=fit_data[ic].ufloat_params,
                    chisq=fit_data[ic].reduced_chisq,
                    quality=quality
                ))
            else:
                quality = 'bad'

            if self.options.return_fit_parameters:
                analysis_results.append(AnalysisResultData(
                    name=PARAMS_ENTRY_PREFIX + self.__class__.__name__ + f'_{op}',
                    value=fit_data[ic],
                    quality=quality,
                ))

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            if scan_qubit == 0:
                xlabel = 'Control siZZle amplitude'
            else:
                xlabel = 'Target siZZle amplitude'

            plotter.set_figure_options(
                xlabel=xlabel,
                ylabel='Hamiltonian components (rad/s)',
                ylim=(-5.e+6, 5.e+6)
            )

            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)

            for ic, label in enumerate(ops):
                plotter.set_series_data(
                    label,
                    x_formatted=amplitudes,
                    y_formatted=unp.nominal_values(components[ic]),
                    y_formatted_err=unp.std_devs(components[ic])
                )

                if fit_data[ic].success:
                    # y_data_with_uncertainty = eval_with_uncertainties(
                    #     x=x_interp,
                    #     model=models[ic],
                    #     params=fit_data.ufloat_params,
                    # )
                    # y_interp = unp.nominal_values(y_data_with_uncertainty)
                    y_interp = fit_results[ic].eval(x=x_interp)

                    plotter.set_series_data(
                        label,
                        x_interp=x_interp,
                        y_interp=y_interp
                    )
                    # if fit_data.covar is not None:
                    #     y_interp_err = unp.std_devs(y_data_with_uncertainty)
                    #     if np.isfinite(y_interp_err).all():
                    #         plotter.set_series_data(
                    #             label,
                    #             y_interp_err=y_interp_err
                    #         )


            figures.append(plotter.figure())

        return analysis_results, figures
