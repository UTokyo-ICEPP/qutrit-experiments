from typing import Any, Iterable, Optional, Sequence, Dict, List, Tuple
import numpy as np
from uncertainties import unumpy as unp

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, Options, BackendTiming
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX

from ...common.util import grounded_gauss_area
from ..hamiltonian_tomography import HamiltonianTomography
from ..qutrit_cr_hamiltonian import QutritCRHamiltonian, QutritCRHamiltonianAnalysis
from ..cr_rabi import cr_rabi_init
from .cr_tomography_postanalysis import compute_omega_0_angle, find_target_angle

twopi = 2. * np.pi


class QutritCXCRTomographyCal(BaseCalibrationExperiment, QutritCRHamiltonian):
    """Qutrit CR Hamiltonian tomography with a CR phase and Stark frequency update."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.calibrate_width = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: List[str] = ['cr', 'cr', 'qutrit_cx_block_phase'],
        cal_parameter_name: List[str] = ['cr_phase', 'stark_frequency', 'omega_zx_sign'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        sign_parameter: str = 'zx'
    ):
        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.
        }
        schedule = calibrations.get_schedule(
            schedule_name[0], physical_qubits, assign_params=assign_params, group=group
        )

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths,
            time_unit=time_unit
        )

        self.sign_parameter = sign_parameter

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group

        # Use arctan(ω0y/ω0x) to determine the CR pulse phase because we know the proper sign of ω0x
        # analytically
        omega_0_angle = compute_omega_0_angle(experiment_data).n
        target_angle = find_target_angle(self.backend, self.physical_qubits)
        current_phase = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits,
                                                       schedule=self._sched_name[0])
        # In circuit notation (left to right):
        #   Pulse(angle=φ) = ShiftPhase(φ) Pulse ShiftPhase(-φ)
        # On IBM backends, ShiftPhase(φ) effects a +φ Z rotation of the state vector on the Bloch
        # sphere (contrary to how rz(φ) is scheduled). Therefore the axis of the polar rotation
        # by Pulse(angle=φ) is rotated -φ from the original Pulse.
        # Thus
        #   omega_0_angle = -current_phase + hardware_offset
        #   target_angle = -new_phase + hardware_offset
        #   -> new_phase = omega_0_angle + current_phase - target_angle
        # tone to point towards target_angle when Pulse(angle=current_phase) points it to
        # omega_0_angle.
        new_phase = omega_0_angle + current_phase - target_angle
        if new_phase > np.pi:
            new_phase -= twopi
        elif new_phase < -np.pi:
            new_phase += twopi

        values = [new_phase]

        if self.experiment_options.calibrate_width:
            # We will calculate the width of the CR pulse that sets the Rabi phase difference of ±π
            # between the 0 and 1 blocks after the CX sequence.
            # The phase-corrected CR pulse corresponds to
            #  CR(±) = blkdiag[V_0(±), V_1(±), V_2(±)]
            # and under the approximation of negligible Z terms
            #  V_j(±) = exp[∓i/2 (ω_jx t + φ_j) x]
            # ignoring the identity terms that will be cancelled by the CX sequence. ω_jx and φ_j
            # can be obtained from the fit results of the child_data of this experiment.
            # The CX sequence
            #  CX = X+ CR(+) X+ CR(+) X+ CR(-)
            # constructs cyclic products of V_js. The V_js commute under the current approximation,
            # so the exponents of the CX unitary are simple linear combinations of the V_j exponents
            # with a composition matrix
            #  C = [[-1, 1, 1], [1, -1, 1], [1, 1, -1]]
            # left-multiplied to the column vector of V_j exponents.
            # Then
            #  CX = exp[-i/2 C @ vec{ω_x τΔ + φ} x].
            # We solve for the minimum τ that satisfies
            #  [C(ωτΔ+φ)]_1x = [C(ωτΔ+φ)]_0x + (π mod 2π).
            #  -> τ = [(π mod 2π) - (C_1 - C_0).φ] / [(C_1 - C_0).ω Δ]

            component_index = experiment_data.metadata["component_child_index"]
            omega = []
            phi = []
            for index in component_index:
                child_data = experiment_data.child_data(index)
                components = child_data.analysis_results('hamiltonian_components').value
                # HamiltonianTomography assumes H = sum_{g=x,y,z} omega^g/2 * g
                omega.append(unp.nominal_values(components) * 2.)

                fit_results = next(res.value for res in child_data.analysis_results()
                                   if res.name.startswith(PARAMS_ENTRY_PREFIX))
                phi.append(fit_results.params['delta']) # 'phi' is the azimuthal angle of the Rabi axis

            omega = np.array(omega)
            phi = np.array(phi)

            # Apply the phase correction
            phase_corr = target_angle - omega_0_angle
            rotation = np.array([[np.cos(phase_corr), -np.sin(phase_corr), 0.],
                                 [np.sin(phase_corr), np.cos(phase_corr), 0.],
                                 [0., 0., 1.]])
            omega = np.tensordot(omega, rotation, (1, 1))

            # Compose the CX combinations
            c1_c0 = np.array([2., -2., 0.])

            cr_width_denom = c1_c0 @ omega[:, 0] * self._backend_data.dt
            cr_width = (np.pi - c1_c0 @ phi) / cr_width_denom
            while cr_width > 0.:
                cr_width -= twopi / np.abs(cr_width_denom)
            while cr_width < 0.:
                cr_width += twopi / np.abs(cr_width_denom)

            cr_rsr = self._cals.get_parameter_value('cr_rsr', self.physical_qubits,
                                                    schedule=self._sched_name[0])
            cr_sigma = self._cals.get_parameter_value('cr_sigma', self.physical_qubits,
                                                      schedule=self._sched_name[0])

            support = cr_width + 2. * cr_sigma * cr_rsr
            cr_duration = BackendTiming(self.backend).round_pulse(samples=support)
            if cr_duration < support:
                cr_duration += self._backend_data.granularity
            cr_margin = cr_duration - support

            values += [cr_width, cr_margin]

        else:
            result = experiment_data.analysis_results('hamiltonian_components', block=False)
            h_components = unp.nominal_values(result.value)

            # Set a tentative Stark frequency depending on the sign of the Iz term
            new_frequency = self._backend_data.drive_freqs[self.physical_qubits[1]]
            if h_components[0, 2] > 0.:
                new_frequency -= 5.e+7
            else:
                new_frequency += 5.e+7

            if self.sign_parameter == 'zx':
                omega_sign = np.sign(h_components[1, 0])
            elif self.sign_parameter == 'ζx':
                omega_sign = np.sign(h_components[2, 0])

            values += [new_frequency, omega_sign]

        for param_name, sched_name, value in zip(self._param_name, self._sched_name, values):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, param_name,
                                            schedule=sched_name, group=group)


class QutritCROmega2ZCal(BaseCalibrationExperiment, HamiltonianTomography):
    """Determination of the counter Stark frequency from the measurement of omega_2z via HT."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: str = 'stark_frequency',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.,
            'cr_stark_amp': 0.,
            'counter_stark_amp': 0.
        }
        schedule = calibrations.get_schedule(
            schedule_name, physical_qubits, assign_params=assign_params, group=group
        )

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            rabi_init=cr_rabi_init(2),
            widths=widths,
            time_unit=time_unit
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group

        result = experiment_data.analysis_results('hamiltonian_components', block=False)

        # Set a tentative Stark frequency depending on the sign of the 2z term
        new_frequency = self._backend_data.drive_freqs[self.physical_qubits[1]]
        if result.value[2].n > 0.:
            new_frequency -= 5.e+7
        else:
            new_frequency += 5.e+7

        BaseUpdater.add_parameter_value(self._cals, experiment_data, new_frequency,
                                        self._param_name, schedule=self._sched_name, group=group)
