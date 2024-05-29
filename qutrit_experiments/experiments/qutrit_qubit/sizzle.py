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
from qiskit_experiments.curve_analysis.utils import convert_lmfit_result, eval_with_uncertainties
from qiskit_experiments.framework import AnalysisResultData, BackendData, ExperimentData, Options
from qiskit_experiments.framework.containers import ArtifactData
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...pulse_library import ModulatedGaussianSquare
from ...util.sizzle import get_qudit_components, sizzle_hamiltonian_shifts, sizzle_shifted_energies
from .spectator_ramsey import SpectatorRamseyXY
from .zzramsey import QutritZZRamsey

twopi = 2. * np.pi


def build_sizzle_schedule(
    frequency: float,
    control: int,
    target: int,
    backend: Backend,
    amplitudes: tuple[float, float],
    angles: tuple[float, float] = (0., 0.),
    control_channel: Optional[pulse.ControlChannel] = None,
    target_channel: Optional[pulse.DriveChannel] = None,
    pre_delay: Optional[int] = None
) -> ScheduleBlock:
    """Build a siZZle schedule using two ModulatedGaussianSquare pulses."""
    backend_data = BackendData(backend)
    target_freq = backend_data.drive_freqs[target]
    detuning = (frequency - target_freq) * backend_data.dt

    if control_channel is None:
        control_channel = backend_data.control_channel((control, target))[0]
    if target_channel is None:
        target_channel = backend_data.drive_channel(target)

    width = Parameter('delay')  # SpectatorRamseyXY requires the parameter to be named delay
    with pulse.build(name='sizzle', default_alignment='left') as sizzle_sched:
        if pre_delay:
            pulse.delay(pre_delay, control_channel)
            pulse.delay(pre_delay, target_channel)

        pulse.play(
            ModulatedGaussianSquare(
                duration=(width + 256),
                amp=amplitudes[0],
                sigma=64,
                freq=detuning,
                width=width,
                angle=angles[0]
            ),
            control_channel
        )
        pulse.play(
            ModulatedGaussianSquare(
                duration=(width + 256),
                amp=amplitudes[1],
                sigma=64,
                freq=detuning,
                width=width,
                angle=angles[1]
            ),
            target_channel
        )

    return sizzle_sched


class SiZZleRamsey(SpectatorRamseyXY):
    """SpectatorRamseyXY experiment for SiZZle."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        frequency: float,
        amplitudes: tuple[float, float],
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        sizzle_schedule = build_sizzle_schedule(frequency, physical_qubits[0], physical_qubits[1],
                                                backend, amplitudes=amplitudes, angles=angles,
                                                control_channel=channels[0],
                                                target_channel=channels[1])
        metadata = {
            'sizzle_frequency': frequency,
            'sizzle_amplitudes': list(amplitudes),
            'sizzle_angles': list(angles)
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        super().__init__(physical_qubits, control_state, delays=delays, osc_freq=osc_freq,
                         delay_schedule=sizzle_schedule, extra_metadata=metadata,
                         backend=backend)


class SiZZleRamseyShift(BatchExperiment):
    """SiZZle + reference SpectatorRamseyXY to compute the frequency shift."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        frequency: float,
        amplitudes: tuple[float, float],
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = [
            SiZZleRamsey(physical_qubits, control_state, frequency,
                         amplitudes=amplitudes, angles=angles, channels=channels, delays=delays,
                         osc_freq=osc_freq, extra_metadata=extra_metadata, backend=backend),
            SpectatorRamseyXY(physical_qubits, control_state, delays=delays, osc_freq=osc_freq,
                              backend=backend)
        ]
        super().__init__(experiments, backend=backend,
                         analysis=SiZZleRamseyShiftAnalysis([exp.analysis for exp in experiments]))


class SiZZleRamseyShiftAnalysis(CompoundAnalysis):
    """Analysis for SiZZleRamseyShift."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata["component_child_index"]
        sizzle_data = experiment_data.child_data(component_index[0])
        shifted_omega = sizzle_data.analysis_results('freq').value * twopi
        ref_data = experiment_data.child_data(component_index[1])
        base_omega = ref_data.analysis_results('freq').value * twopi
        analysis_results.append(
            AnalysisResultData(name='omega_zs', value=shifted_omega - base_omega)
        )
        return analysis_results, figures


class SiZZle(QutritZZRamsey):
    """QutritZZRamsey experiment for SiZZle."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        amplitudes: tuple[float, float],
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        sizzle_schedule = build_sizzle_schedule(frequency, physical_qubits[0], physical_qubits[1],
                                                backend, amplitudes=amplitudes, angles=angles,
                                                control_channel=channels[0],
                                                target_channel=channels[1])
        metadata = {
            'sizzle_frequency': frequency,
            'sizzle_amplitudes': list(amplitudes),
            'sizzle_angles': list(angles)
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        super().__init__(physical_qubits, delays=delays, osc_freq=osc_freq,
                         delay_schedule=sizzle_schedule, extra_metadata=metadata,
                         backend=backend)


class SiZZleShift(BatchExperiment):
    """SiZZle + reference SpectatorRamseyXY to compute the frequency shifts."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        amplitudes: tuple[float, float],
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = [
            SiZZle(physical_qubits, frequency,
                   amplitudes=amplitudes, angles=angles, channels=channels, delays=delays,
                   osc_freq=osc_freq, extra_metadata=extra_metadata, backend=backend),
            QutritZZRamsey(physical_qubits, delays=delays, osc_freq=osc_freq,
                           extra_metadata=extra_metadata, backend=backend)
        ]
        super().__init__(experiments, backend=backend,
                         analysis=SiZZleShiftAnalysis([exp.analysis for exp in experiments]))


class SiZZleShiftAnalysis(CompoundAnalysis):
    """Analysis for SiZZleShift."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata["component_child_index"]
        sizzle_data = experiment_data.child_data(component_index[0])
        shifted_omegas = sizzle_data.analysis_results('omega_zs').value
        ref_data = experiment_data.child_data(component_index[1])
        base_omegas = ref_data.analysis_results('omega_zs').value
        analysis_results.append(
            AnalysisResultData(name='omega_zs', value=shifted_omegas - base_omegas)
        )
        return analysis_results, figures


class SiZZleFrequencyScan(BatchExperiment):
    """Frequency scan of SiZZle."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.frequencies_of_interest = {}
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Sequence[float],
        amplitudes: tuple[float, float],
        measure_shift: bool = True,
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []
        for freq in frequencies:
            exp = SiZZle(physical_qubits, frequency=freq, amplitudes=amplitudes, angles=angles,
                         channels=channels, delays=delays, osc_freq=osc_freq, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        if measure_shift:
            exp = QutritZZRamsey(physical_qubits, delays=delays, osc_freq=osc_freq, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=SiZZleFrequencyScanAnalysis(analyses))
        self.measure_shift = measure_shift

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['frequencies_of_interest'] = dict(self.experiment_options.frequencies_of_interest)
        metadata['measure_shift'] = self.measure_shift
        hvars = self._backend.configuration().hamiltonian['vars']
        keys = sum(([f'wq{qubit}', f'delta{qubit}', f'omegad{qubit}']
                    for qubit in self.physical_qubits), [])
        keys += [f'jq{min(self.physical_qubits)}q{max(self.physical_qubits)}']
        metadata['hvars'] = {key: hvars[key] for key in keys}
        return metadata


class SiZZleFrequencyScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZleFrequencyScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.base_omegas = np.zeros(3)
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Plot Iz, zz, and ζz components as functions of siZZle frequency."""
        component_index = experiment_data.metadata["component_child_index"]
        if experiment_data.metadata['measure_shift']:
            ref_data = experiment_data.child_data(component_index[-1])
            base_omegas = ref_data.analysis_results('omega_zs').value
            component_index = list(component_index[:-1])
        else:
            base_omegas = self.options.base_omegas

        num_freqs = len(component_index)
        frequencies = np.empty(num_freqs)
        components = np.empty((3, num_freqs), dtype=object)

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            frequencies[ichild] = child_data.metadata['sizzle_frequency']
            components[:, ichild] = child_data.analysis_results('omega_zs').value - base_omegas

        analysis_results.extend([
            AnalysisResultData(name='scan_frequencies', value=frequencies),
            AnalysisResultData(name='omega_zs', value=components)
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='siZZle frequency (Hz)',
                ylabel='Hamiltonian components (rad/s)'
            )
            for ic, label in enumerate(['Iz', 'zz', 'ζz']):
                plotter.set_series_data(
                    label,
                    x_formatted=np.array(frequencies),
                    y_formatted=unp.nominal_values(components[ic]),
                    y_formatted_err=unp.std_devs(components[ic])
                )

            freqs_fine = np.linspace(frequencies[0], frequencies[-1], 100)
            amps = experiment_data.child_data(component_index[0]).metadata['sizzle_amplitudes']
            if experiment_data.metadata['measure_shift']:
                prediction = sizzle_hamiltonian_shifts(
                    experiment_data.metadata['hvars'],
                    experiment_data.metadata['physical_qubits'],
                    amps,
                    freqs_fine
                )
            else:
                prediction = get_qudit_components(
                    sizzle_shifted_energies(
                        experiment_data.metadata['hvars'],
                        experiment_data.metadata['physical_qubits'],
                        amps,
                        freqs_fine
                    ),
                    truncate_to=(3, 2)
                )

            for ic, label in enumerate(['Iz', 'zz', 'ζz']):
                plotter.set_series_data(
                    label,
                    x_interp=freqs_fine,
                    y_interp=prediction[:, ic, 1]
                )

            figure = plotter.figure()
            ax = figure.axes[0]
            ax.axhline(0., linestyle='--')
            for label, freq in experiment_data.metadata['frequencies_of_interest'].items():
                ax.axvline(freq, label=label)

            figures.append(figure)

        return analysis_results, figures


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
        amplitudes: tuple[float, float],
        measure_shift: bool = True,
        control_phase_offsets: Optional[Sequence[float]] = None,
        target_angle: float = 0.,
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
                         angles=(offset, target_angle), channels=channels, delays=delays,
                         osc_freq=osc_freq, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        if measure_shift:
            exp = QutritZZRamsey(physical_qubits, delays=delays, osc_freq=osc_freq, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=SiZZlePhaseScanAnalysis(analyses))

        if control_phase_offsets is not None:
            self.set_experiment_options(phase_offsets=control_phase_offsets)

        self.measure_shift = measure_shift

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['measure_shift'] = self.measure_shift
        return metadata


class SiZZlePhaseScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZlePhaseScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.base_omegas = np.zeros(3)
        options.expected_signs = [-1, 1, -1]  # Iz, zz, ζz
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Plot Iz, zz, and ζz components as functions of siZZle phase offset."""
        component_index = experiment_data.metadata["component_child_index"]
        if experiment_data.metadata['measure_shift']:
            ref_data = experiment_data.child_data(component_index[-1])
            base_omegas = ref_data.analysis_results('omega_zs').value
            component_index = list(component_index[:-1])
        else:
            base_omegas = self.options.base_omegas

        num_phases = len(component_index)
        phase_offsets = np.empty(num_phases)
        components = np.empty((3, num_phases), dtype=object)

        ops = ['Iz', 'zz', 'ζz']

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            phase_offsets[ichild] = -np.diff(child_data.metadata['sizzle_angles'])
            components[:, ichild] = child_data.analysis_results('omega_zs').value - base_omegas

        models = [
            lmfit.models.ExpressionModel(
                expr=f'amp_{label} * cos(x + cr_phase_offset) + base_{label}'
            ) for label in ops
        ]
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
            analysis_results.append(ArtifactData(name='fit_summary', data=fit_data))

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
    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequency: float,
        amplitudes: Union[tuple[Sequence[float], float], tuple[float, Sequence[float]]],
        measure_shift: bool = True,
        angles: tuple[float, float] = (0., 0.),
        channels: tuple[pulse.ControlChannel, pulse.DriveChannel] = (None, None),
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        try:
            len(amplitudes[0])
        except TypeError:
            scan_qubit = 1
        else:
            scan_qubit = 0

        experiments = []
        analyses = []

        amp_argument = [None] * 2
        amp_argument[1 - scan_qubit] = amplitudes[1 - scan_qubit]

        for amplitude in amplitudes[scan_qubit]:
            amp_argument[scan_qubit] = amplitude
            extra_metadata = {'scan_qubit': scan_qubit}
            exp = SiZZle(physical_qubits, frequency=frequency, amplitudes=amp_argument,
                         angles=angles, channels=channels, delays=delays, osc_freq=osc_freq,
                         extra_metadata=extra_metadata, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        if measure_shift:
            exp = QutritZZRamsey(physical_qubits, delays=delays, osc_freq=osc_freq, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=SiZZleAmplitudeScanAnalysis(analyses))
        self.measure_shift = measure_shift

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['measure_shift'] = self.measure_shift
        return metadata


class SiZZleAmplitudeScanAnalysis(CompoundAnalysis):
    """Analysis for SiZZleAmplitudeScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        options.base_omegas = np.zeros(3)
        options.curve_fit = True
        options.return_fit_parameters = False
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        """Plot Iz, zz, and ζz components as functions of siZZle amplitude."""
        component_index = experiment_data.metadata["component_child_index"]
        if experiment_data.metadata['measure_shift']:
            ref_data = experiment_data.child_data(component_index[-1])
            base_omegas = ref_data.analysis_results('omega_zs').value
            component_index = list(component_index[:-1])
        else:
            base_omegas = self.options.base_omegas

        num_amps = len(component_index)
        amplitudes = np.empty(num_amps)
        components = np.empty((3, num_amps), dtype=object)

        ops = ['Iz', 'zz', 'ζz']

        for ichild, idx in enumerate(component_index):
            child_data = experiment_data.child_data(idx)
            scan_qubit = child_data.metadata['scan_qubit']
            amplitudes[ichild] = child_data.metadata['sizzle_amplitudes'][scan_qubit]
            components[:, ichild] = child_data.analysis_results('omega_zs').value - base_omegas
            if ichild == 0:
                fixed_amplitude = child_data.metadata['sizzle_amplitudes'][1 - scan_qubit]

        analysis_results.extend([
            AnalysisResultData(name='scan_amplitudes', value=amplitudes),
            AnalysisResultData(name='omega_zs', value=components)
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            if scan_qubit == 0:
                xlabel = 'Control siZZle amplitude'
            else:
                xlabel = 'Target siZZle amplitude'

            plotter.set_figure_options(
                xlabel=xlabel,
                ylabel='Hamiltonian components (rad/s)'
            )

            for ic, op in enumerate(ops):
                plotter.set_series_data(
                    op,
                    x_formatted=amplitudes,
                    y_formatted=unp.nominal_values(components[ic]),
                    y_formatted_err=unp.std_devs(components[ic])
                )

        if not self.options.curve_fit:
            figures.append(plotter.figure())
            return analysis_results, figures

        fit_results = {}
        fit_data = {}
        for ic, op in enumerate(ops):
            xval = amplitudes
            yval = unp.nominal_values(components[ic])

            if op == 'Iz':
                if scan_qubit == 0:
                    model = lmfit.models.ConstantModel()
                    c = np.mean(yval)
                    params = model.make_params(c=c)
                else:
                    model = lmfit.models.QuadraticModel()
                    diff_amp = np.diff(xval)
                    a = np.mean(np.diff(np.diff(yval) / diff_amp) / diff_amp[:-1] / 2.)
                    c = yval[0] - a * np.square(xval[0])
                    params = model.make_params(a=a, b={'value': 0., 'vary': False}, c=c)
            else:
                if scan_qubit == 1 and fixed_amplitude == 0.:
                    model = lmfit.models.ConstantModel()
                    c = np.mean(yval)
                    params = model.make_params(c=c)
                else:
                    model = lmfit.models.LinearModel()
                    slope = (yval[-1] - yval[0]) / (xval[-1] - xval[0])
                    intercept = yval[0] - slope * xval[0]
                    params = model.make_params(slope=slope, intercept=intercept)

            fit_results[op] = model.fit(yval, params, x=xval)
            fit_data[op] = convert_lmfit_result(fit_results[op], [model], xval, yval)

            if fit_data[op].success:
                quality = 'good'

                analysis_results.append(AnalysisResultData(
                    name=f'ω_{op}_coeffs',
                    value=fit_data[op].ufloat_params,
                    chisq=fit_data[op].reduced_chisq,
                    quality=quality
                ))
            else:
                quality = 'bad'

            analysis_results.append(ArtifactData(name='fit_summary', data=fit_data[op]))

        if self.options.plot:
            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)

            for op in ops:
                if fit_data[op].success:
                    # y_data_with_uncertainty = eval_with_uncertainties(
                    #     x=x_interp,
                    #     model=models[ic],
                    #     params=fit_data.ufloat_params,
                    # )
                    # y_interp = unp.nominal_values(y_data_with_uncertainty)
                    y_interp = fit_results[op].eval(x=x_interp)

                    plotter.set_series_data(
                        op,
                        x_interp=x_interp,
                        y_interp=y_interp
                    )
                    # if fit_data.covar is not None:
                    #     y_interp_err = unp.std_devs(y_data_with_uncertainty)
                    #     if np.isfinite(y_interp_err).all():
                    #         plotter.set_series_data(
                    #             op,
                    #             y_interp_err=y_interp_err
                    #         )

            figures.append(plotter.figure())

        return analysis_results, figures
