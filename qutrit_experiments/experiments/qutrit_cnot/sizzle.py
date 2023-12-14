import logging
from typing import Any, Dict, Iterable, List, Tuple, Optional, Sequence
import numpy as np
import scipy.optimize as sciopt
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import ExperimentData

from ..sizzle_1z import SiZZle1ZPhaseScan
from ..qutrit_cr_hamiltonian import QutritCRHamiltonian
from ...calibrations.sizzle import sizzle_hamiltonian_shifts

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class QutritCRSiZZlePhaseCal(BaseCalibrationExperiment, SiZZle1ZPhaseScan):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: Optional[List[str]] = None,
        auto_update: bool = True,
        group: str = 'default',
        control_phase_offsets: Optional[Sequence[float]] = None,
        amplitudes: Optional[Tuple[float, float]] = None,
        delays: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None,
    ):
        if cal_parameter_name is None:
            cal_parameter_name = ['cr_phase', 'cr_stark_phase']

        cr_sched_channels = calibrations.get_schedule(schedule_name, physical_qubits).channels
        control_channel = next(ch for ch in cr_sched_channels if isinstance(ch, pulse.ControlChannel))
        target_channel = next(ch for ch in cr_sched_channels if isinstance(ch, pulse.DriveChannel))
        sizzle_frequency = calibrations.get_parameter_value('stark_frequency', physical_qubits,
                                                            schedule=schedule_name)

        hvars = backend.configuration().hamiltonian['vars']
        max_shifts = sizzle_hamiltonian_shifts(hvars, physical_qubits, (0.1, 0.1), sizzle_frequency)
        max_shift_1z = max_shifts[0, 1] - max_shifts[1, 1] + max_shifts[2, 1]
        # Determine the delay durations from the maximum possible frequency
        max_duration = int(np.round(np.pi / np.abs(max_shift_1z) / backend.dt / 16)) * 16
        max_duration = min(512, max_duration)
        delay_durations = [max_duration]
        for factor in [0.6, 0.25]:
            duration = int(np.round(max_duration * factor / 16)) * 16
            if duration == 0:
                continue
            delay_durations.insert(0, duration)
        delay_durations.insert(0, 0)

        super().__init__(
            calibrations,
            physical_qubits,
            sizzle_frequency,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            control_phase_offsets=control_phase_offsets,
            amplitudes=amplitudes,
            channels=(control_channel, target_channel),
            delay_durations=delay_durations,
            num_points=10,
            backend=backend
        )

        omega_amp = (-max_shifts[1, 1] + max_shifts[2, 1]) * backend.dt
        self.analysis.set_options(
            p0={
                'omega_amp': omega_amp,
                'omega_base': max_shifts[0, 1] * backend.dt
            },
            bounds={
                'omega_amp': (0., np.inf) if omega_amp > 0. else (-np.inf, 0.),
                'channel_phase_diff': (-np.pi, np.pi)
            }
        )


    def _attach_calibrations(self, circuit: QuantumCircuit):
        # We can in principle attach the x calibration for the control qubit, but that would change nothing
        pass

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group,
            ) for pname in self._param_name
        ]
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group

        channel_phase_diff = BaseUpdater.get_value(experiment_data, 'channel_phase_diff')

        # This is the hardware-specific phase difference between the control and target drive lines
        # Set the overall CR phase as well as the Stark phase
        for pname in self._param_name:
            BaseUpdater.add_parameter_value(self._cals, experiment_data, channel_phase_diff,
                                            pname, schedule=self._sched_name, group=group)


class QutritCRSiZZleFrequencyCal(BaseCalibrationExperiment, QutritCRHamiltonian):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: Optional[List[str]] = None,
        auto_update: bool = True,
        group: str = 'default',
        widths: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None
    ):
        if cal_parameter_name is None:
            cal_parameter_name = ['stark_frequency', 'cr_stark_amp', 'cr_stark_phase', 'counter_stark_amp']

        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.,
            'cr_stark_amp': 0.,
            'counter_stark_amp': 0.
        }
        schedule = calibrations.get_schedule(
            'cr', physical_qubits, assign_params=assign_params
        )

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            widths=widths,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = list(
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group,
            ) for pname in self._param_name
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        h_components = experiment_data.analysis_results('hamiltonian_components', block=False).value
        h_components = unp.nominal_values(h_components)

        # Frequency search range is determined by the maximum modulation frequency
        hvars = self.backend.configuration().hamiltonian['vars']
        qubits = self.physical_qubits

        max_mod_freq = twopi / (16. * self.backend.dt)
        min_freq = hvars[f'wq{qubits[1]}'] - max_mod_freq
        max_freq = hvars[f'wq{qubits[1]}'] + max_mod_freq
        frequencies = np.linspace(min_freq, max_freq, 100)

        # Filter out unusable frequencies
        resonances = np.sort(list(hvars[f'wq{q}'] for q in qubits)
                             + list(hvars[f'wq{q}'] + hvars[f'delta{q}'] for q in qubits))
        near_resonance = np.any((frequencies[:, None] > resonances[None, :] - 2.e+7 * twopi)
                                & (frequencies[:, None] < resonances[None, :] + 2.e+7 * twopi),
                                axis=1)

        # We don't know the final cr_width yet, but it will be order a couple hundred
        min_mod_freq = twopi / (256. * self.backend.dt)
        unexpressible_lower = hvars[f'wq{qubits[1]}'] - min_mod_freq
        unexpressible_upper = hvars[f'wq{qubits[1]}'] + min_mod_freq
        small_mod_freq = np.any((frequencies[:, None] > unexpressible_lower)
                                & (frequencies[:, None] < unexpressible_upper),
                                axis=1)

        frequencies = frequencies[np.logical_not(near_resonance | small_mod_freq)]

        test_amp = 0.1
        component_shifts = sizzle_hamiltonian_shifts(hvars, qubits, (test_amp, test_amp),
                                                     frequencies)

        # Find the best amplitude combination at each frequency
        def residual_z_at_freq(params, ifreq):
            rac, rat = params
            shifts = component_shifts[ifreq]
            # Assuming Iz shift is quadratic in target amp and zz & zetaz shifts are prop to control and
            # target amps
            return (np.square(h_components[0, 2] + shifts[0, 1] * (rat ** 2))
                    + np.sum(np.square(h_components[1:, 2] + shifts[1:, 1] * rat * rac)))

        min_res = np.full(frequencies.shape + (3,), np.inf)
        for ifreq in range(len(frequencies)):
            res = sciopt.minimize(residual_z_at_freq, [1., 1.], (ifreq,),
                                  bounds=[(-3., 3.), (0., 3.)])
            if res.success:
                min_res[ifreq, 0] = res.fun
                min_res[ifreq, 1:] = res.x * test_amp

        # Frequency that minimizes the residual z
        best_ifreq = np.argmin(min_res[:, 0])

        # Find the true optimum
        freq_init = frequencies[best_ifreq]

        freq_region = np.searchsorted(resonances, freq_init)
        if freq_region < len(resonances):
            freq_upper_bound = min(resonances[freq_region], max_freq)
            if freq_init < unexpressible_lower:
                freq_upper_bound = min(freq_upper_bound, unexpressible_lower)

            if freq_region > 0:
                freq_lower_bound = max(resonances[freq_region - 1], min_freq)
                if freq_init > unexpressible_upper:
                    freq_lower_bound = max(freq_lower_bound, unexpressible_upper)
            else:
                freq_lower_bound = min_freq
        else:
            freq_upper_bound = max_freq
            freq_lower_bound = max(unexpressible_upper, resonances[-1])

        def residual_z(params):
            rfreq, ac, at = params
            shifts = sizzle_hamiltonian_shifts(hvars, qubits, (ac, at), rfreq * freq_init)
            return np.sum(np.square(h_components[:, 2] + shifts[:, 1]))

        x0 = [1.] + list(min_res[best_ifreq, 1:])

        res = sciopt.minimize(residual_z, x0,
                              bounds=[(freq_lower_bound, freq_upper_bound), (-0.3, 0.3), (0., 0.3)])

        if not res.success:
            logger.warning('Could not find a good sizzle frequency')
            return

        group = self.experiment_options.group
        current_phase = self._cals.get_parameter_value(self._param_name[2], qubits,
                                                       schedule=self._sched_name)
        values = [
            res.x[0] * freq_init / twopi,
            np.abs(res.x[1]),
            current_phase + (0. if res.x[1] > 0. else np.pi),
            res.x[2]
        ]
        for pname, value in zip(self._param_name, values):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=self._sched_name, group=group)


class QutritCRSiZZleAmplitudeCal(BaseCalibrationExperiment, QutritCRHamiltonian):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        free_hamiltonian: np.ndarray,
        schedule_name: str = 'cr',
        cal_parameter_name: Optional[List[str]] = None,
        auto_update: bool = True,
        group: str = 'default',
        widths: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None
    ):
        if cal_parameter_name is None:
            cal_parameter_name = ['cr_stark_amp', 'counter_stark_amp']

        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.
        }
        schedule = calibrations.get_schedule(
            'cr', physical_qubits, assign_params=assign_params
        )

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            widths=widths,
            backend=backend
        )

        self.free_hamiltonian = np.array(free_hamiltonian)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = list(
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group,
            ) for pname in self._param_name
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        h_components = experiment_data.analysis_results('hamiltonian_components', block=False).value
        h_components = unp.nominal_values(h_components)

        delta_h = h_components - self.free_hamiltonian

        def objective(params):
            cscale, tscale = params
            scale_arr = np.array([cscale ** 2, cscale * tscale, cscale * tscale])
            return np.sum(np.square(delta_h[:, 2] * scale_arr + self.free_hamiltonian))

        res = sciopt.minimize(objective, [1., 1.], bounds=[(0., None), (0., None)])

        if not res.success:
            logger.warning('Failed to find a good siZZle amplitude combination')
            return

        group = self.experiment_options.group

        for pname, scale in zip(self._param_name, res.x):
            current_amp = self._cals.get_parameter_value(pname, self.physical_qubits,
                                                         schedule=self._sched_name)

            BaseUpdater.add_parameter_value(self._cals, experiment_data, current_amp * scale, pname,
                                            schedule=self._sched_name, group=group)
