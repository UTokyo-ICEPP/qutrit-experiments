"""Standard fine frequency calibration with error amplification."""

from collections.abc import Sequence
from typing import Optional, Union
import lmfit
import numpy as np
from uncertainties import unumpy as unp

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.library import FineFrequency
import qiskit_experiments.curve_analysis as curve

from ..transpilation import map_to_physical_qubits
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..experiment_mixins.map_to_physical_qubits import MapToPhysicalQubits


class EFFineFrequency(MapToPhysicalQubits, EFSpaceExperiment, FineFrequency):
    """Standard fine frequency calibration with error amplification."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delay_duration: int,
        backend: Optional[Backend] = None,
        repetitions: Optional[list[int]] = None,
    ):
        super().__init__(physical_qubits, delay_duration, backend=backend, repetitions=repetitions)

        self.analysis = EFFineFrequencyAnalysis()
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2
            },
            p0={'phase_offset': 0.}
        )

        # phase_offset p0 must be set due to a bug in ErrorAmplificationAnalysis
        # Line 152
        #  phi = user_opt.p0.get("phase_offset", fixed_params.get("phase_offset", 0.0))
        # Because user_opt.p0 is always filled with the full parameter keys with None
        # as the initial value, we end up with phi = None unless explicitly set here

    def circuits(self) -> list[QuantumCircuit]:
        circuits = super().circuits()

        for circuit in circuits:
            for inst in circuit.data:
                if inst.operation.name == 'rz':
                    inst.operation.name = 'rz12'
                elif inst.operation.name == 'sx':
                    inst.operation.name = 'sx12'

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                      self._backend_data.coupling_map)


class EFFineFrequencyAnalysis(curve.ErrorAmplificationAnalysis):
    """Error amplification analysis with cosine envelope."""

    @staticmethod
    def modulated_cosine(x, amp, d_theta, beat_freq, phase_offset, base, angle_per_gate):
        angular_freq = d_theta + angle_per_gate
        angular_beat_freq = 2. * np.pi * beat_freq
        return (0.5 * amp * unp.cos(angular_freq * x - phase_offset)
                * unp.cos(angular_beat_freq * x)) + base

    @classmethod
    def _default_options(cls):
        r"""Return the default analysis options.

        See :meth:`~qiskit_experiment.curve_analysis.CurveAnalysis._default_options` for
        descriptions of analysis options.

        Analysis Options:
            max_good_angle_error (float): The maximum angle error for which the fit is
                considered as good. Defaults to :math:`\pi/2`.
        """
        default_options = super()._default_options()
        default_options.result_parameters.append("beat_freq")
        return default_options

    def __init__(self, name: Optional[str] = None):
        super(curve.ErrorAmplificationAnalysis, self).__init__(
            models=[lmfit.Model(self.modulated_cosine, name='modulated_cosine')],
            name=name
        )

    def _initialize(self, experiment_data: ExperimentData):
        super()._initialize(experiment_data)

        if self.options.fixed_parameters.get('angle_per_gate', 0.) == 0.:
            raise ValueError('EFFineFrequencyAnalysis must have nonzero angle_per_gate')

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        """Create algorithmic guess with analysis options and curve data.

        Args:
            user_opt: Fit options filled with user provided guess and bounds.
            curve_data: Formatted data collection to fit.

        Returns:
            list of fit options that are passed to the fitter function.
        """
        options = super()._generate_fit_guesses(user_opt, curve_data)

        base = options[0].p0['base']

        normy_ = np.abs(curve_data.y - base)
        normy_ /= np.amax(normy_)

        imax = np.argmax(normy_)

        apg = self.options.fixed_parameters.get("angle_per_gate")

        window = np.round(2. * np.pi / apg).astype(int)

        # move from imax in both directions and search for a point where the max
        # within the window starts increasing
        for direction in (1, -1):
            cursor = imax + direction - window // 2
            prev_max = normy_[imax]
            found_minimum = False

            while 0 <= cursor < normy_.size - window:
                window_max = np.amax(normy_[cursor:cursor + window])
                if window_max > prev_max:
                    cursor -= direction
                    beat_phase = np.pi / 2.
                    found_minimum = True
                    break

                prev_max = window_max
                cursor += direction
            else:
                # if no minimum was found, estimate the current beat phase
                beat_phase = np.arccos(normy_[cursor])

            xdiff = (curve_data.x[cursor] - curve_data.x[imax]) * direction
            beat_freq = beat_phase / xdiff / (2. * np.pi)

            if found_minimum:
                break

        for option in options:
            option.p0.set_if_empty(beat_freq=beat_freq)

        return options

    def _evaluate_quality(self, fit_data: curve.FitData) -> Union[str, None]:
        """Algorithmic criteria for whether the fit is good or bad.

        A good fit has:
            - a reduced chi-squared lower than three,
            - a measured angle error that is smaller than the allowed maximum good angle error.
              This quantity is set in the analysis options.
        """
        quality = super()._evaluate_quality(fit_data)

        if quality == 'bad':
            return quality

        fit_beat_freq = fit_data.fitval("beat_freq")
        ac_freq = self.options.fixed_parameters.get('angle_per_gate')

        # beat_freq is beat phase per gate; should be smaller than angle_per_gate
        criteria = [
            abs(fit_beat_freq.nominal_value) < abs(ac_freq)
        ]

        if all(criteria):
            return "good"

        return "bad"


class EFFineFrequencyCal(BaseCalibrationExperiment, EFFineFrequency):
    """Calibration experiment for EFFineFrequency."""
    @classmethod
    def _default_experiment_options(cls):
        """default values for the fine frequency calibration experiment.

        Experiment Options:
            dt (float): The duration of the time unit ``dt`` of the delay and schedules in seconds.
        """
        options = super()._default_experiment_options()
        options.dt = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "f12",
        auto_update: bool = True,
        delay_duration: Optional[int] = None,
        repetitions: list[int] = None
    ):
        r"""see class :class:`FineFrequency` for details.
        Note that this class implicitly assumes that the target angle of the gate is :math:`\pi/2`
        as seen from the default analysis options. This experiment can be seen as a calibration of a
        finite duration ``rz(pi/2)`` gate with any error attributed to a frequency offset in the
        qubit.
        Args:
            qubit: The qubit for which to run the fine frequency calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            delay_duration: The duration of the delay at :math:`n=1`. If this value is
                not given then the duration of the gate named ``gate_name`` in the
                calibrations will be used.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
            gate_name: This argument is only needed if ``delay_duration`` is None. This
                should be the name of a valid schedule in the calibrations.
        """
        if delay_duration is None:
            delay_duration = calibrations.get_schedule('sx12', physical_qubits).duration

        super().__init__(
            calibrations,
            physical_qubits,
            delay_duration,
            schedule_name=None,
            cal_parameter_name=cal_parameter_name,
            backend=backend,
            auto_update=auto_update,
            repetitions=repetitions,
        )

        if self.backend is not None:
            self.set_experiment_options(dt=self._backend_data.dt)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> dict[str, any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
            delay_duration: The duration of the first delay.
            dt: The number of ``dt`` units of the delay.
        """
        metadata = super()._metadata()
        metadata["delay_duration"] = self.experiment_options.delay_duration
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            group=self.experiment_options.group,
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group
        prev_freq = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            group=group
        )
        tau = self.experiment_options.delay_duration
        dt = self.experiment_options.dt

        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)
        new_freq = prev_freq + d_theta / (2 * np.pi * tau * dt)

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_freq, self._param_name, group=group
        )
