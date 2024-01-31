"""Qutrit-qubit HT to make a very rough estimate of what the CR width for CRCR will be."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import BackendTiming, ExperimentData

from ...util.pulse_area import grounded_gauss_area
from ..qutrit_cr_hamiltonian import QutritCRHamiltonianTomography
from .util import RCRType

twopi = 2. * np.pi


class CycledRepeatedCRRoughWidthCal(BaseCalibrationExperiment, QutritCRHamiltonianTomography):
    """Qutrit-qubit HT to make a very rough estimate of what the CR width for CRCR will be."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'cr_base_angle', 'rcr_type'],
        schedule_name: str = ['cr', 'cr', None],
        auto_update: bool = True,
        widths: Optional[Sequence[float]] = None,
        time_unit: Optional[float] = None,
    ):
        assign_params = {cal_parameter_name[0]: Parameter('width'), 'margin': 0}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)
        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            param_name=cal_parameter_name,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            widths=widths,
            time_unit=time_unit
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        omegas = unp.nominal_values(
            experiment_data.analysis_results('hamiltonian_components', block=False).value
        )
        # Calculate the angle overshoot
        angle = np.arctan2(omegas[0, 1], omegas[0, 0])
        current_angle = self._cals.get_parameter_value(self._param_name[1], self.physical_qubits,
                                                       self._sched_name[1])
        cr_base_angle = (current_angle - angle) % twopi

        # Adjust the CR pulse angle to eliminate IY
        rot_matrix = np.array([
            [np.cos(angle), np.sin(angle), 0.],
            [-np.sin(angle), np.cos(angle), 0.],
            [0., 0., 1.]
        ])
        omegas = rot_matrix @ omegas
        # Control-basis omega
        omegas_cb = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]) @ omegas

        # Approximate angular rate at which the Rabi phase difference accummulates between 0/2 and
        # 1 blocks
        omega_x = omegas_cb[:, 0]
        crcr_omega_0 = 2. * np.array([omega_x[2], omega_x[0]])
        crcr_omega_1 = 2. * np.sum(omega_x[None, :] * np.array([[1., 1., -1.], [-1., 1., 1.]]),
                                   axis=1)
        crcr_rel_freqs = np.abs(crcr_omega_1 - crcr_omega_0)
        # Whichever RCR type with larger angular rate will be used
        rcr_type_index = np.argmax(crcr_rel_freqs)
        rcr_type = [RCRType.X, RCRType.X12][rcr_type_index]
        # Compute the width accounting for the gaussiansquare flanks
        sigma = self._cals.get_parameter_value('sigma', self.physical_qubits, self._sched_name[0])
        rsr = self._cals.get_parameter_value('rsr', self.physical_qubits, self._sched_name[0])
        flank = grounded_gauss_area(sigma, rsr, True)
        cr_width = BackendTiming(self._backend).round_pulse(
            samples=np.pi / crcr_rel_freqs[rcr_type_index] / self._backend.dt - flank
        )

        for pname, sname, value in zip(self._param_name, self._sched_name,
                                       [cr_width, cr_base_angle, rcr_type]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=sname,
                group=self.experiment_options.group
            )
