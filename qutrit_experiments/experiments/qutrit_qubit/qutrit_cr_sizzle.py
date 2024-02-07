"""UT scan over target Stark amplitudes to minimize the Iz component of the unitary."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import ExperimentData

from .qutrit_qubit_tomography import QutritQubitTomographyScan


class QutritCRTargetStarkCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """UT scan over target Stark amplitudes to minimize the Iz component of the unitary."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'counter_stark_amp',
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        counter_stark_amp = Parameter(cal_parameter_name)
        assign_params = {cal_parameter_name: counter_stark_amp}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        circuit = QuantumCircuit(2)
        circuit.append(Gate('cr', 2, [counter_stark_amp]), [0, 1])
        circuit.add_calibration('cr', physical_qubits, schedule, [counter_stark_amp])

        if amplitudes is None:
            amplitudes = np.linspace(0.005, 0.16, 10)

        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            param_name=cal_parameter_name,
            values=amplitudes,
            measure_preparations=measure_preparations,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        amplitudes = [experiment_data.child_data(idx).metadata[self._param_name]
                      for idx in component_index]
        unitaries = experiment_data.analysis_results('unitary_parameters', block=False).value
        unitaries = unp.nominal_values(unitaries)
        theta_iz = np.einsum('oc,sck->sok',
                             np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                             unitaries)[:, 0, 2]
        best_amp = amplitudes[np.argmin(np.abs(theta_iz))]
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, best_amp, self._param_name, schedule=self._sched_name,
            group=self.experiment_options.group
        )


class QutritCRControlStarkCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """UT scan over control Stark amplitudes to minimize the zz & ζz components of the unitary."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['cr_stark_amp', 'cr_stark_sign_phase'],
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        parameters = [Parameter(pname) for pname in cal_parameter_name]
        assign_params = {p.name: p for p in parameters}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        circuit = QuantumCircuit(2)
        circuit.append(Gate('cr', 2, parameters), [0, 1])
        circuit.add_calibration('cr', physical_qubits, schedule, parameters)

        if amplitudes is None:
            max_amp = 0.99 - calibrations.get_parameter_value('cr_amp', physical_qubits,
                                                              schedule_name)
            amplitudes = np.linspace(-max_amp, max_amp, 16)

        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            param_name=cal_parameter_name[0],
            values=amplitudes,
            measure_preparations=measure_preparations,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            angle_param_name=cal_parameter_name[1]
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        amplitudes = [experiment_data.child_data(idx).metadata[self._param_name[0]]
                      for idx in component_index]
        unitaries = experiment_data.analysis_results('unitary_parameters', block=False).value
        unitaries = unp.nominal_values(unitaries)
        theta_zs = np.einsum('oc,sck->sok',
                             np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                             unitaries)[:, 1:, 2]
        best_amp = amplitudes[np.argmin(np.sum(np.square(theta_zs), axis=-1))]
        phase = 0. if best_amp > 0. else np.pi
        for pname, value in zip(self._param_name, [best_amp, phase]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )

