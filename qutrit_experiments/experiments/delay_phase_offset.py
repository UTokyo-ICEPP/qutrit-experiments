"""Ramsey experiment with constant delay."""

from collections.abc import Callable, Sequence
from typing import Any, Optional, Union
import numpy as np
import lmfit
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import RZGate, SXGate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import BaseExperiment, ExperimentData, Options

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..experiment_mixins.map_to_physical_qubits import MapToPhysicalQubits
from ..gates import RZ12Gate, SX12Gate
from ..util.dummy_data import ef_memory, single_qubit_counts

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
     ┌───────────┐┌─────┐┌─────┐┌────┐
     ┤ SX^{(t_0)}├┤Rz(φ)├┤Rz(ε)├┤ SX ├
     └───────────┘└─────┘└─────┘└────┘
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
     ┌───────────┐┌─────┐┌─────────────┐┌────┐
     ┤ SX^{(t_1)}├┤Rz(φ)├┤Rz(ω_z τ + ε)├┤ SX ├
     └───────────┘└─────┘└─────────────┘└────┘
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
    __sx_gate__ = SXGate
    __rz_gate__ = RZGate

    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()

        options.delay_durations = [0, 128, 160, 192, 240]
        options.delay_schedule = None
        options.num_points = 16
        options.pre_schedule = None
        # Call Rz(-phi) when Rz(phi) is desired so that the instruction in the schedule reads
        # shift_phase(-phi), which corresponds to the physical Rz(phi)
        options.invert_phase_sign = True
        options.dummy_omega_z = 1.e+5
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

        template.append(self.__sx_gate__(), qargs)

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

        template.append(self.__rz_gate__(phase_shift), qargs)
        template.append(self.__sx_gate__(), qargs)
        template.measure(self.target_logical_qubit, 0)

        template.metadata = {
            'experiment_type': self._type,
            'qubits': self.physical_qubits,
            'readout_qubits': [self.physical_qubits[self.target_logical_qubit]]
        }
        if self.experiment_index is not None:
            template.metadata['experiment_index'] = self.experiment_index

        circuits = []
        for delay_value in delay_durations:
            for phase_value in phase_shifts:
                assign_params = {delay: delay_value, phase_shift: phase_value}
                if self.experiment_options.invert_phase_sign:
                    assign_params[phase_shift] *= -1.

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

    def dummy_data(
        self,
        transpiled_circuits: list[QuantumCircuit]
    ) -> list[Union[np.ndarray, Counts]]:
        return self._dummy_data((0, 1))

    def _dummy_data(self, states: tuple[int, int]) -> list[np.ndarray]:
        phase_shifts = np.linspace(0., twopi, self.experiment_options.num_points, endpoint=False)
        delay_durations = self.experiment_options.delay_durations
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        num_qubits = 1
        meas_return = self.run_options.get('meas_return', 'avg')

        amp = 0.49
        base = 0.51

        data = []

        for delay_value in delay_durations:
            offset = (self.experiment_options.dummy_omega_z * twopi * delay_value
                      * self._backend_data.dt)
            p_ground = -amp * np.cos(phase_shifts + offset) + base

            # IQClassification -> meas_level==KERNELED
            if self.run_options.meas_level == MeasLevel.KERNELED:
                data += ef_memory(p_ground, shots, num_qubits,
                                  meas_return=meas_return, states=states)
            else:
                data += single_qubit_counts(p_ground, shots, num_qubits)

        return data


class EFRamseyPhaseSweep(EFSpaceExperiment, RamseyPhaseSweep):
    """RamseyPhaseSweep for EF space."""
    __sx_gate__ = SX12Gate
    __rz_gate__ = RZ12Gate
    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        return self._dummy_data((0, 2))


class RamseyPhaseSweepAnalysis(curve.CurveAnalysis):
    """Analysis for RamseyPhaseSweep."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.nwind_hypotheses = list(range(-2, 3))
        return options

    def __init__(self, delay_durations: Sequence[int]):
        super().__init__(models=[
            lmfit.models.ExpressionModel(
                expr=f'amp * cos(x + epsilon + omega_z_dt * {delay}) + base',
                name=f'delay{delay}'
            ) for delay in delay_durations
        ])

        self.set_options(
            result_parameters=['omega_z_dt'],
            normalization=False,
            outcome='1',
            data_subfit_map={f'delay{delay}': {'delay': delay} for delay in delay_durations},
            bounds={
                'amp': (0., 0.6),
                'base': (0., 1.),
                'epsilon': (-np.pi, np.pi)
            }
        )
        self.plotter.set_figure_options(
            xlabel="Phase shift",
            ylabel="$P_e$"
        )

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
        options = []
        if user_opt.p0.get('epsilon') is not None or user_opt.p0.get('omega_z_dt') is not None:
            options.append(user_opt)

        data_0 = curve_data.get_subset_of('delay0')
        base = np.mean(data_0.y)
        amp = np.sqrt(np.mean(np.square(data_0.y - base)) * 2.)
        cos_epsilon = np.mean(data_0.y * np.cos(data_0.x)) / amp * 2.
        sin_epsilon = -np.mean(data_0.y * np.sin(data_0.x)) / amp * 2.
        epsilon = np.arctan2(sin_epsilon, cos_epsilon)

        delay_durations = sorted(s['delay'] for s in self.options.data_subfit_map.values())
        min_delay = delay_durations[1]
        data_1 = curve_data.get_subset_of(f'delay{min_delay}')
        cos_offset = np.mean(data_1.y * np.cos(data_1.x)) / amp * 2.
        sin_offset = -np.mean(data_1.y * np.sin(data_1.x)) / amp * 2.
        offset_diff = np.arctan2(sin_offset, cos_offset) - epsilon

        # Account for mod(2pi)s of offset_diff
        for nwind in self.options.nwind_hypotheses:
            new_opt = user_opt.copy()
            new_opt.p0.update(amp=amp, base=base, epsilon=epsilon,
                              omega_z_dt=(offset_diff + twopi * nwind) / min_delay)
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
