"""Fine-grain calibration of the DRAG beta parameter for X12 and SX12 gates."""
from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import pulse, QuantumCircuit
from qiskit.circuit import CircuitInstruction, Gate
from qiskit.circuit.library import RZGate, SXGate
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.library import FineDrag

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..gates import RZ12Gate, SX12Gate
from ..transpilation import map_to_physical_qubits
from ..util.dummy_data import ef_memory, single_qubit_counts


class EFFineDrag(EFSpaceExperiment, FineDrag):
    """FineDrag experiment for the 1<->2 space Rx pulses.

    The original FineDrag uses sx and rz gates in its circuits. We replace them, in a rather hacky
    way, with sx12 and rz12 to avoid having to rewrite largely identical code. Note that this can
    cause problems if the 0<->1 space sx and/or rz gates are also to be used for some reason.

    Also note that sandwiching the x12 gate with Rz12(pi) is wrong in principle, as that leaves a
    geometric phase between |0> and |1>-|2> spaces. The correct sequence for a inverted-sign Rx gate
    is Rz12(-pi).Rx.Rz12(pi). We are saved by the fact that the experiment operates entirely in the
    |1>-|2> space, rendering the geometric phase global.
    """
    def circuits(self) -> list[QuantumCircuit]:
        circuits = super().circuits()

        for circuit in circuits:
            for inst in circuit.data:
                if isinstance(inst.operation, RZGate):
                    inst.operation = RZ12Gate(inst.operation.params[0])
                elif isinstance(inst.operation, SXGate):
                    inst.operation = SX12Gate()

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        repetitions = self.experiment_options.repetitions
        gate = self.experiment_options.gate

        circuits = self.circuits()
        first_circuit = map_to_physical_qubits(circuits[0], self.physical_qubits,
                                               self._backend_data.coupling_map)

        transpiled_circuits = [first_circuit]
        rep_position = next(pos for pos, inst in enumerate(first_circuit.data)
                            if isinstance(inst.operation, RZ12Gate)) - 1

        for circuit, repetition in zip(circuits[1:], repetitions[1:]):
            tcirc = first_circuit.copy()
            for _ in range(repetition - repetitions[0]):
                # inserting at the same position -> reverse order
                qubits = [self.physical_qubits[0]]
                tcirc.data.insert(rep_position, CircuitInstruction(RZ12Gate(-np.pi), qubits))
                tcirc.data.insert(rep_position, CircuitInstruction(gate, qubits))
                tcirc.data.insert(rep_position, CircuitInstruction(RZ12Gate(np.pi), qubits))
                tcirc.data.insert(rep_position, CircuitInstruction(gate, qubits))

            tcirc.calibrations = circuit.calibrations
            tcirc.metadata = circuit.metadata

            transpiled_circuits.append(tcirc)

        return transpiled_circuits

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = np.full(len(self.experiment_options.repetitions), 0.5)
        num_qubits = 1

        if self.run_options.meas_level == MeasLevel.KERNELED:
            if self._final_xgate:
                states = (0, 2)
            else:
                states = (1, 2)

            meas_return=self.run_options.get('meas_return', 'avg')

            return ef_memory(one_probs, shots, num_qubits, meas_return,
                             states=states)
        return single_qubit_counts(one_probs, shots, num_qubits)


class EFFineDragCal(BaseCalibrationExperiment, EFFineDrag):
    """Calibration experiment for EFFineDrag."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        """Default experiment options.
        Experiment Options:
            target_angle (float): The target rotation angle of the gate being calibrated.
                This value is needed for the update rule.
        """
        options = super()._default_experiment_options()
        options.target_angle = np.pi
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "beta",
        auto_update: bool = True,
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            Gate(name=schedule_name, num_qubits=1, params=[]),
            schedule_name=schedule_name,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()

        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            self._sched_name,
            self.experiment_options.group
        )
        metadata["cal_param_name"] = self._param_name
        metadata["cal_schedule"] = self._sched_name
        metadata["target_angle"] = self.experiment_options.target_angle
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the drag parameter of the pulse in the calibrations."""

        result_index = self.experiment_options.result_index
        group = self.experiment_options.group
        target_angle = self.experiment_options.target_angle
        qubits = self.physical_qubits

        schedule = self._cals.get_schedule(self._sched_name, qubits)

        # Obtain sigma as it is needed for the fine DRAG update rule.
        sigmas = []
        for block in schedule.blocks:
            if isinstance(block, pulse.Play) and hasattr(block.pulse, "sigma"):
                sigmas.append(getattr(block.pulse, "sigma"))

        if len(set(sigmas)) != 1:
            raise CalibrationError(
                "Cannot run fine Drag calibration on a schedule with multiple values of sigma."
            )

        if len(sigmas) == 0:
            raise CalibrationError(f"Could not infer sigma from {schedule}.")

        d_theta = BaseUpdater.get_value(experiment_data, "d_theta", result_index)

        # See the documentation in fine_drag.py for the derivation of this rule.
        d_beta = -np.sqrt(np.pi) * d_theta * sigmas[0] / target_angle**2
        old_beta = experiment_data.metadata["cal_param_value"]
        new_beta = old_beta + d_beta

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_beta, self._param_name, schedule, group
        )


class EFFineXDragCal(EFFineDragCal):
    """Specialization of EFFineDragCal for X12."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "beta",
        auto_update: bool = True,
    ):
        r"""see class :class:`FineDrag` for details.
        Args:
            qubit: The qubit for which to run the fine drag calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            cal_parameter_name: The name of the parameter in the schedule to update.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name="x12",
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        super()._attach_calibrations(circuit)
        schedule = self._cals.get_schedule('x12', self.physical_qubits[0])
        circuit.add_calibration('x12', [self.physical_qubits[0]], schedule)


class EFFineSXDragCal(EFFineDragCal):
    """Specialization of EFFineDragCal for SX12."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "beta",
        auto_update: bool = True,
    ):
        r"""see class :class:`FineDrag` for details.
        Args:
            qubit: The qubit for which to run the fine drag calibration.
            calibrations: The calibrations instance with the schedules.
            backend: Optional, the backend to run the experiment on.
            cal_parameter_name: The name of the parameter in the schedule to update.
            auto_update: Whether or not to automatically update the calibrations. By
                default this variable is set to True.
        """
        super().__init__(
            physical_qubits,
            calibrations,
            schedule_name="sx12",
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
        )

    @staticmethod
    def _pre_circuit() -> QuantumCircuit:
        circuit = QuantumCircuit(1)
        circuit.append(SX12Gate(), [0])
        return circuit

    def _attach_calibrations(self, circuit: QuantumCircuit):
        schedule = self._cals.get_schedule('sx12', self.physical_qubits[0])
        circuit.add_calibration('sx12', [self.physical_qubits[0]], schedule)
