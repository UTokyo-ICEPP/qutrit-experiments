"""Error amplification experiment to fine-calibrate the CR width and Rx amplitude."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.options import Options
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis
from qiskit_experiments.framework import BaseExperiment

from ...experiment_mixins import MapToPhysicalQubits
from .util import RCRType, get_margin, make_crcr_circuit


class CycledRepeatedCRFine(MapToPhysicalQubits, BaseExperiment):
    """Error amplification experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.repetitions = np.arange(1, 9)

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: RCRType,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
        self._cr_schedules = tuple(cr_schedules)
        self._rx_schedule = rx_schedule
        self._rcr_type = rcr_type

        if repetitions is None:
            self.set_experiment_options(repetitions=repetitions)

        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2,
            }
        )

    def circuits(self) -> list[QuantumCircuit]:
        crcr_circuit = make_crcr_circuit(self._physical_qubits, self._cr_schedules,
                                         self._rx_schedule, self._rcr_type)

        circuits = []

        for add_x in [0, 1]:
            circuit = QuantumCircuit(2, 1)
            self._prep_circuit(circuit)
            if add_x == 1:
                circuit.x(1)
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': add_x,
                'series': 'spam-cal'
            }
            circuits.append(circuit)

        for repetition in self.experiment_options.repetitions:
            circuit = QuantumCircuit(2, 1)
            self._prep_circuit(circuit)
            circuit.sx(1)
            for _ in range(repetition):
                circuit.compose(crcr_circuit, inplace=True)
                self._iteration_circuit(circuit)
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': repetition,
                'series': 1
            }
            circuits.append(circuit)

        return circuits

    def _prep_circuit(self, circuit: QuantumCircuit):
        pass

    def _iteration_circuit(self, circuit: QuantumCircuit):
        pass


class CycledRepeatedCRFineCal(BaseCalibrationExperiment, CycledRepeatedCRFine):
    """Base calibration experiment for CycledRepeatedCRFine."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = '',
        schedule_name: str = '',
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        cr_schedules = [calibrations.get_schedule('cr', physical_qubits)]
        # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
        assign_params = {pname: np.pi for pname in
                         ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']}
        cr_schedules.append(calibrations.get_schedule('cr', physical_qubits,
                                                      assign_params=assign_params))
        rx_schedule = calibrations.get_schedule('offset_rx', physical_qubits[1])
        
        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            rx_schedule,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            repetitions=repetitions
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        result_index = self.experiment_options.result_index
        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)
        self._update_calibrations_from_d_theta(self, experiment_data, d_theta)

    def _update_calibrations_from_d_theta(self, experiment_data: ExperimentData, d_theta: float):
        pass


class CycledRepeatedCRFineRxAmp(CycledRepeatedCRFine):
    """Pingpong experiment on control=0."""
    def _iteration_circuit(self, circuit: QuantumCircuit):
        # Create an artificial oscillation
        circuit.x(1)


class CycledRepeatedCRFineRxAmpCal(CycledRepeatedCRFineCal, CycledRepeatedCRFineRxAmp):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        angle_per_amp: float,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['amp', 'sign_angle'],
        schedule_name: str = 'offset_rx',
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        super().__init__(physical_qubits, calibrations, backend=backend,
                         cal_parameter_name=cal_parameter_name, schedule_name=schedule_name,
                         repetitions=repetitions, auto_update=auto_update)
        self.angle_per_amp = angle_per_amp

    def _update_calibrations_from_d_theta(self, experiment_data: ExperimentData, d_theta: float):
        current_amp = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits[1],
                                                     schedule=self._sched_name)
        sign_angle = self._cals.get_parameter_value(self._param_name[1], self.physical_qubits[1],
                                                    schedule=self._sched_name)
        if sign_angle != 0.:
            current_amp *= -1.
        
        amp = current_amp - d_theta / self.angle_per_amp
        sign_angle = 0. if new_amp > 0. else np.pi
        for pname, value in zip(self._param_name, [amp, sign_angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )


class CycledRepeatedCRFineCRWidth(CycledRepeatedCRFine):
    def _prep_circuit(self, circuit: QuantumCircuit):
        circuit.x(0)


class CycledRepeatedCRFineCRWidthCal(CycledRepeatedCRFineCal, CycledRepeatedCRFineCRWidth):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        angle_per_dt: float,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin'],
        schedule_name: str = 'cr',
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        super().__init__(physical_qubits, calibrations, backend=backend,
                         cal_parameter_name=cal_parameter_name, schedule_name=schedule_name,
                         repetitions=repetitions, auto_update=auto_update)
        self.angle_per_dt = angle_per_dt

    def _update_calibrations_from_d_theta(self, experiment_data: ExperimentData, d_theta: float):
        current_width = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits[1],
                                                       schedule=self._sched_name)
        width = current_width - d_theta / self.angle_per_dt
        null_sched = self._cals.get_schedule(self._sched_name, self.physical_qubits,
                                             assign_params={p: 0. for p in self._param_name})
        margin = get_margin(null_sched.duration, width, self._backend)

        for pname, value in zip(self._param_name, [width, margin]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )
