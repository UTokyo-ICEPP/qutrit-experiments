"""Error amplification experiment to fine-calibrate the CR width and Rx amplitude."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers.options import Options
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework.experiment_data import ExperimentData
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis
from qiskit_experiments.framework import BaseExperiment

from ...experiment_mixins import MapToPhysicalQubits
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import X12Gate
from .util import RCRType, get_margin, make_crcr_circuit


class CycledRepeatedCRPingPong(MapToPhysicalQubits, BaseExperiment):
    """Error amplification experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.repetitions = np.arange(1, 9)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: RCRType,
        cx_sign: Optional[Number] = None,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
        self._control_state = control_state
        self._crcr_circuit = make_crcr_circuit(physical_qubits, cr_schedules, rx_schedule, rcr_type)
        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)

        # Fit function is P(1) = A/2*cos(n*(apg + dθ) - po) + B
        # -> With an initial SX, phase offset is always π/2 for control=0/2 but dependent on the CX
        # sign for control=1
        if control_state == 1 and cx_sign is None:
            raise RuntimeError('cx_sign is required for control_state=1')
        
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2 * cx_sign,
            },
            outcome='1'
        )

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []

        for add_x in [0, 1]:
            circuit = QuantumCircuit(2, 1)
            if self._control_state != 0:
                circuit.x(0)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0])
            if add_x == 1:
                circuit.x(1)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0]) # restore qubit space
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': add_x,
                'series': 'spam-cal'
            }
            circuits.append(circuit)

        for repetition in self.experiment_options.repetitions:
            circuit = QuantumCircuit(2, 1)
            if self._control_state != 0:
                circuit.x(0)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0])
            circuit.sx(1)
            for _ in range(repetition):
                circuit.compose(self._crcr_circuit, inplace=True)
                if self._control_state != 1:
                    circuit.x(1)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0]) # restore qubit space
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': repetition,
                'series': 1
            }
            circuits.append(circuit)

        return circuits
    

class CycledRepeatedCRFine(BatchExperiment):
    """Perform CycledRepeatedCRPingPong for control state 0 and 1 to calibrate width and rx amp."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: RCRType,
        cx_sign: Number,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = [
            CycledRepeatedCRPingPong(physical_qubits, ic, cr_schedules, rx_schedule, rcr_type,
                                     cx_sign=cx_sign, repetitions=repetitions, backend=backend)
            for ic in range(2)
        ]
        super().__init__(experiments, backend=backend)


class CycledRepeatedCRFineCal(BaseCalibrationExperiment, CycledRepeatedCRFine):
    """Calibration experiment for CycledRepeatedCRFine."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.calibration_qubit_index = {(pname, 'offset_rx'): [1]
                                           for pname in ['amp', 'sign_angle']}
        return options
    
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        width_rate_params: np.ndarray,
        amp_rate: float,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin', 'amp', 'sign_angle'],
        schedule_name: list[str] = ['cr', 'cr', 'offset_rx', 'offset_rx'],
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
            calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            repetitions=repetitions
        )
        self.width_rate_params = np.array(width_rate_params)
        self.amp_rate = amp_rate

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        d_thetas = np.array([
            experiment_data.child_data(idx).analysis_results('d_theta').value.n
            for idx in component_index
        ])
        # dθ_i = a_i dw + b_i + c dA
        # -> (dw, dA)^T = ([a_0 c], [a_1 c])^{-1} (dθ_0 - b_0, dθ_1 - b_1)^T
        mat = np.stack([self.width_rate_params[:, 0], np.full(2, self.amp_rate)], axis=1)
        d_width, d_amp = np.linalg.inv(mat) @ (d_thetas - self.width_rate_params[:, 1])

        # Calculate the new width
        current_width = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits,
                                                       schedule=self._sched_name[0])
        width = current_width - d_width
        assign_params = {p: 0. for p in self._param_name[:2]}
        null_width_sched = self._cals.get_schedule(self._sched_name[0], self.physical_qubits,
                                                   assign_params=assign_params)
        margin = get_margin(null_width_sched.duration, width, self._backend)

        for pname, sname, value in zip(self._param_name[:2], self._sched_name[:2], [width, margin]):
            param_value = ParameterValue(
                value=value,
                date_time=BaseUpdater._time_stamp(experiment_data),
                group=self.experiment_options.group,
                exp_id=experiment_data.experiment_id,
            )
            self._cals.add_parameter_value(param_value, pname, self.physical_qubits, sname)
                                           
        # Calculate the new Rx amplitude
        current_amp = self._cals.get_parameter_value(self._param_name[2], self.physical_qubits[1],
                                                     schedule=self._sched_name[2])
        sign_angle = self._cals.get_parameter_value(self._param_name[3], self.physical_qubits[1],
                                                    schedule=self._sched_name[3])
        if sign_angle != 0.:
            current_amp *= -1.

        amp = current_amp - d_amp
        sign_angle = 0. if amp > 0. else np.pi
        for pname, sname, value in zip(self._param_name[2:], self._sched_name[2:],
                                       [abs(amp), sign_angle]):
            param_value = ParameterValue(
                value=value,
                date_time=BaseUpdater._time_stamp(experiment_data),
                group=self.experiment_options.group,
                exp_id=experiment_data.experiment_id,
            )
            self._cals.add_parameter_value(param_value, pname, self.physical_qubits[1], sname)
       