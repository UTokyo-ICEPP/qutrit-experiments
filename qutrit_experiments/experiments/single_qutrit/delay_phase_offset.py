"""Ramsey experiment with constant delay."""

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union
import numpy as np
import lmfit
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import BaseExperiment, ExperimentData, Options

from ...experiment_mixins.ef_space import EFSpaceExperiment
from ...experiment_mixins.map_to_physical_qubits import MapToPhysicalQubits

twopi = 2. * np.pi


class RamseyPhaseSweep(MapToPhysicalQubits, BaseExperiment):
    r"""SX + delay + phase shift + SX experiment for measuring a static Z Hamiltonian.

    This experiment can be used to measure e.g. the static ZZ interaction as well as to calibrate
    the drive frequency.

    Let the current drive frequency be :math:`\omega_q + \delta \omega`, where :math:`\omega_q` is
    the qubit (qutrit) frequency. The SX pulse at time :math:`t` corresponds to

    ... math::

        SX^{(t)} = R_z(-\delta \omega t) SX R_z(\delta \omega t)

    so the excited-state probability after the "prompt" sequence
     ┌────┐┌─────┐┌─────┐┌────────────┐
     ┤ SX ├┤Rz(ε)├┤Rz(φ)├┤ SX^{(t_0)} ├
     └────┘└─────┘└─────┘└────────────┘
    where t_0 and ε are the duration and phase shift of the schedule (0 for a standard
    delay experiment), is

    ... math::

        P_e^{p} & = \lvert \langle 1 \rvert R_z(-\delta\omega t_0) \sqrt{X} R_z(\delta\omega t_0)
                            R_z(\phi) R_z(\epsilon) \sqrt{X} \lvert 0 \rangle \rvert^2 \\
                & = \frac{1}{4} \lvert (-i \langle 0 \rvert + \langle 1 \rvert)
                                       R_z(\delta\omega t_0 + \epsilon + \phi)
                                       (\lvert 0 \rangle - i \lvert 1 \rangle) \rvert^2 \\
                & = \cos^2 \left(\frac{1}{2} [\delta\omega t_0 + \epsilon + \phi]\right) \\
                & = \frac{1}{2} \cos (\phi + \delta\omega t_0 + \epsilon) + \frac{1}{2}.

    On the other hand, assuming a Hamiltonian

    ... math::

        H = \frac{\omega_z}{2} Z,

    the corresponding probability from the "delayed" sequence with delay τ
     ┌────┐┌─────────────┐┌─────┐┌────────────┐
     ┤ SX ├┤Rz(ω_z τ + ε)├┤Rz(φ)├┤ SX^{(t_1)} ├
     └────┘└─────────────┘└─────┘└────────────┘
    with t_1 = t_0 + τ, is

    ... math::

        P_e^{d} = \frac{1}{2} \cos (\phi + \delta\omega t_0 + \epsilon
                                    + [\delta\omega + \omega_z] \tau) + \frac{1}{2}.

    Performing the standard OscillationAnalysis for both experiments, the fit parameter ``phase``
    will correspond to

    ... math::

        \Phi^{p} = \delta\omega t_0 + \epsilon

    and

    ... math::

        \Phi^{d} = \delta\omega t_0 + \epsilon + (\delta\omega + \omega_z) \tau,

    respectively, and therefore the phase offset difference :math:`\Delta\Phi` is

    ... math::

        \Delta\Phi = \Phi^{d} - \Phi^{p} = (\delta\omega + \omega_z) \tau.

    For a frequency calibration experiment, we don't apply any active Hamiltonian enhancement
    (:math:`\omega_z = 0`), and measure only :math:`\delta\omega` from the phase offset difference.
    When the same measurement is performed with the neighboring qubit in different excitations, the
    measured :math:`\delta\omega` values can be used to calculate the static ZZ interaction. If
    we apply an off-resonant tone of a given duration instead of the delay, we can measure the
    resulting Stark shift :math:`\omega_z`.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()

        options.delay_durations = [0, 128, 160, 192, 240]
        options.delay_schedule = None
        options.num_points = 16
        options.pre_schedule = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delay_durations: Optional[Sequence[int]] = None,
        delay_schedule: Optional[ScheduleBlock] = None,
        num_points: Optional[int] = None,
        pre_schedule: Optional[Union[ScheduleBlock, tuple[Gate, Sequence[int]], Callable]] = None,
        target_logical_qubit: int = 0,
        extra_metadata: Optional[dict[str, Any]] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        if (durations_exp := delay_durations) is None:
            durations_exp = self._default_experiment_options().delay_durations

        super().__init__(physical_qubits, analysis=RamseyPhaseSweepAnalysis(durations_exp),
                         backend=backend)

        self.set_experiment_options(
            delay_schedule=delay_schedule,
            pre_schedule=pre_schedule
        )
        if delay_durations is not None:
            self.set_experiment_options(delay_durations=delay_durations)
        if num_points is not None:
            self.set_experiment_options(num_points=num_points)

        self.target_logical_qubit = target_logical_qubit

        self.extra_metadata = {} if extra_metadata is None else dict(extra_metadata)
        self.experiment_index = experiment_index

    def _append_pre_gate(self, circuit) -> None:
        pre_schedule = self.experiment_options.pre_schedule
        if isinstance(pre_schedule, (pulse.Schedule, pulse.ScheduleBlock)):
            circuit.append(Gate(pre_schedule.name, 1, []), [self.target_logical_qubit])
            physical_qubit = self.physical_qubits[self.target_logical_qubit]
            circuit.add_calibration(pre_schedule.name, [physical_qubit], pre_schedule)
        elif isinstance(pre_schedule, Sequence):
            # (Gate, qargs)
            circuit.append(*pre_schedule)
        elif callable(pre_schedule):
            pre_schedule(circuit)

        circuit.barrier()

    def circuits(self) -> list[QuantumCircuit]:
        delay_schedule = self.experiment_options.delay_schedule
        delay_durations = self.experiment_options.delay_durations
        phase_shifts = np.linspace(0., twopi, self.experiment_options.num_points, endpoint=False)

        delay = Parameter('delay')
        phase_shift = Parameter('phase_shift')
        template = QuantumCircuit(len(self.physical_qubits), 1)

        self._append_pre_gate(template)

        qargs = [self.target_logical_qubit]

        template.sx(self.target_logical_qubit)

        if delay_schedule is None:
            template.delay(delay, qargs)
        else:
            template.append(Gate('delay_gate', 1, [delay]), qargs)

            parameter = delay_schedule.get_parameters('delay')[0]
            template.add_calibration('delay_gate',
                                     [self.physical_qubits[self.target_logical_qubit]],
                                     delay_schedule.assign_parameters({parameter: delay},
                                                                      inplace=False),
                                     params=[delay])

        template.rz(phase_shift, self.target_logical_qubit)
        template.sx(self.target_logical_qubit)
        template.measure(self.target_logical_qubit, 0)

        template.metadata = {}
        if self.experiment_index is not None:
            template.metadata['experiment_index'] = self.experiment_index

        circuits = []
        for delay_value in delay_durations:
            for phase_value in phase_shifts:
                assign_params = {delay: delay_value, phase_shift: phase_value}
                circuit = template.assign_parameters(assign_params, inplace=False)
                circuit.metadata.update(xval=phase_value, delay=delay_value)

                circuits.append(circuit)

        return circuits

    def _metadata(self):
        metadata = super()._metadata()
        metadata['dt'] = self._backend.dt
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata


class EFRamseyPhaseSweep(EFSpaceExperiment, RamseyPhaseSweep):
    """RamseyPhaseSweep for EF space."""


class RamseyPhaseSweepAnalysis(curve.CurveAnalysis):
    """Analysis for RamseyPhaseSweep."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        # Whether to use a common sine amplitude for the fit. Value=True reduces the ndof of the
        # fit, but the amplitude may vary between delays if e.g. relaxation is a major issue
        options.common_amp = True
        options.nwind_hypotheses = list(range(-2, 3))
        return options

    def __init__(self, delay_durations: Sequence[int]):
        super().__init__()

        self.set_options(
            result_parameters=['omega_z_dt'],
            normalization=False,
            outcome='1',
            data_subfit_map={f'delay{delay}': {'delay': delay} for delay in delay_durations},
            bounds={'epsilon': (-np.pi, np.pi)}
        )
        self.plotter.set_figure_options(
            xlabel="Phase shift",
            ylabel="$P_e$"
        )

    def _initialize(self, experiment_data: ExperimentData):
        delay_durations = sorted(m['delay'] for m in self.options.data_subfit_map.values())

        if self.options.common_amp:
            self._models = [
                lmfit.models.ExpressionModel(
                    expr=f'amp * cos(x + epsilon + omega_z_dt * {delay}) + base',
                    name=f'delay{delay}'
                ) for delay in delay_durations
            ]
            self.options.bounds.update({
                'amp': (0., 0.6),
                'base': (0., 1.)
            })
        else:
            self._models = [
                lmfit.models.ExpressionModel(
                    expr=f'amp{delay} * cos(x + epsilon + omega_z_dt * {delay}) + base{delay}',
                    name=f'delay{delay}'
                ) for delay in delay_durations
            ]
            for delay in delay_durations:
                self.options.bounds.update({
                    f'amp{delay}': (0., 0.6),
                    f'base{delay}': (0., 1.)
                })

        super()._initialize(experiment_data)

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.ScatterTable,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        """Create algorithmic initial fit guess from analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            list of fit options that are passed to the fitter function.
        """
        options = []
        if user_opt.p0.get('epsilon') is not None or user_opt.p0.get('omega_z_dt') is not None:
            options.append(user_opt)

        def get_data_base_amp(delay):
            data = curve_data.filter(series=f'delay{delay}')
            base = np.mean(data.y)
            amp = np.sqrt(np.mean(np.square(data.y - base)) * 2.)
            return data, base, amp

        data_0, base_0, amp_0 = get_data_base_amp(0)
        cos_epsilon = np.mean(data_0.y * np.cos(data_0.x)) / amp_0 * 2.
        sin_epsilon = -np.mean(data_0.y * np.sin(data_0.x)) / amp_0 * 2.
        epsilon = np.arctan2(sin_epsilon, cos_epsilon)

        delay_durations = sorted(s['delay'] for s in self.options.data_subfit_map.values())
        min_delay = delay_durations[1]
        data_1, base_1, amp_1 = get_data_base_amp(min_delay)
        cos_offset = np.mean(data_1.y * np.cos(data_1.x)) / amp_1 * 2.
        sin_offset = -np.mean(data_1.y * np.sin(data_1.x)) / amp_1 * 2.
        offset_diff = np.arctan2(sin_offset, cos_offset) - epsilon

        user_opt.p0.update(epsilon=epsilon)

        if self.options.common_amp:
            user_opt.p0.update(amp=amp_0, base=base_0)
        else:
            user_opt.p0.update({'amp0': amp_0, 'base0': base_0,
                                f'amp{min_delay}': amp_1, f'base{min_delay}': base_1})
            for delay in delay_durations[2:]:
                _, base, amp = get_data_base_amp(delay)
                user_opt.p0.update({f'amp{delay}': amp, f'base{delay}': base})

        # Account for mod(2pi)s of offset_diff
        for nwind in self.options.nwind_hypotheses:
            new_opt = user_opt.copy()
            new_opt.p0.update(omega_z_dt=(offset_diff + twopi * nwind) / min_delay)
            options.append(new_opt)

        return options


class EFRamseyPhaseSweepFrequencyCal(BaseCalibrationExperiment, EFRamseyPhaseSweep):
    """Frequency calibration with RamseyPhaseSweep."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        delay_durations: Optional[Sequence[int]] = None,
        num_points: Optional[int] = None,
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=None,
            cal_parameter_name='f12',
            auto_update=auto_update,
            delay_durations=delay_durations,
            num_points=num_points,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        prev_freq = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            group=group
        )

        omega_z = (BaseUpdater.get_value(experiment_data, 'omega_z_dt', result_index)
                   / self._backend_data.dt)
        new_freq = prev_freq - omega_z / twopi

        BaseUpdater.add_parameter_value(self._cals, experiment_data, new_freq, self._param_name,
                                        group=group)
