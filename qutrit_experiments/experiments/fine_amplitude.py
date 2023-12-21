"""Fine amplitude calibration for X12 and SX12 gates."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import FineAmplitude, FineAmplitudeCal

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..gates import X12Gate, SX12Gate
from ..transpilation import map_to_physical_qubits
from ..util.dummy_data import ef_memory, single_qubit_counts


class CustomTranspiledFineAmplitude(FineAmplitude):
    """FineAmplitude with optimized transpiler sequence."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        repetitions = self.experiment_options.repetitions
        gate = self.experiment_options.gate
        circuits = self.circuits()
        icirc = 0

        transpiled_circuits = []

        if self.experiment_options.add_cal_circuits:
            transpiled_circuits.extend(
                map_to_physical_qubits(circuit, self.physical_qubits,
                                       self._backend_data.coupling_map)
                for circuit in circuits[:2]
            )
            icirc = 2

        first_circuit = map_to_physical_qubits(circuits[icirc], self.physical_qubits,
                                               self._backend_data.coupling_map)
        transpiled_circuits.append(first_circuit)

        gate_position = next(pos for pos, inst in enumerate(first_circuit.data)
                             if inst.operation.name == gate.name)

        for circuit, repetition in enumerate(circuits[icirc + 1:], repetitions[1:]):
            tcirc = first_circuit.copy()
            for _ in range(repetition - repetitions[0]):
                tcirc.data.insert(gate_position,
                                  CircuitInstruction(gate, [self.physical_qubits[0]]))
            tcirc.metadata = circuit.metadata
            transpiled_circuits.append(tcirc)

        return transpiled_circuits


class EFFineAmplitude(EFSpaceExperiment, CustomTranspiledFineAmplitude):
    """EF-space FineAmplitude."""
    def _spam_cal_circuits(self, meas_circuit: QuantumCircuit) -> list[QuantumCircuit]:
        cal_circuits = []

        qubits = meas_circuit.get_instructions("measure")[0][1]

        for add_x in [0, 1]:
            circ = QuantumCircuit(self.num_qubits, meas_circuit.num_clbits)

            if add_x:
                for qubit in qubits:
                    circ.append(X12Gate(), qargs=[qubit])

            circ.compose(meas_circuit, inplace=True)

            circ.metadata = {
                "xval": add_x,
                "series": "spam-cal",
                "qubits": self.physical_qubits
            }

            cal_circuits.append(circ)

        return cal_circuits


class EFFineAmplitudeCal(FineAmplitudeCal, EFFineAmplitude):
    """Calibration experiment for EFFineAmplitude."""
    def _attach_calibrations(self, circuit: QuantumCircuit):
        for gate in ['x12', 'sx12']:
            schedule = self._cals.get_schedule(gate, self.physical_qubits)
            circuit.add_calibration(gate, self.physical_qubits, schedule)


class EFFineXAmplitudeCal(EFFineAmplitudeCal):
    """Specialization of EFFineAmplitude for X12."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            'x12',
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            gate=X12Gate()
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2,
            }
        )

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        """The preparation circuit is an sx gate to move to the equator of the Bloch sphere."""
        circuit = super()._pre_circuit(num_clbits)
        circuit.append(SX12Gate(), qargs=[0])
        return circuit

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        repetitions = np.array(self.experiment_options.repetitions)
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = 0.5 * np.cos(np.pi / 2. + 0.02 + (np.pi + 0.01) * repetitions) + 0.5
        num_qubits = 1

        if self.experiment_options.add_cal_circuits:
            one_probs = np.concatenate(([1., 0.], one_probs))

        if self.run_options.meas_level == MeasLevel.KERNELED:
            if self._final_xgate:
                states = (0, 2)
            else:
                states = (1, 2)

            meas_return=self.run_options.get('meas_return', 'avg')

            return ef_memory(one_probs, shots, num_qubits, meas_return,
                             states=states)
        return single_qubit_counts(one_probs, shots, num_qubits)


class EFFineSXAmplitudeCal(EFFineAmplitudeCal):
    """Specialization of EFFineAmplitude for SX12."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.
        Experiment Options:
            add_sx (bool): This option is False by default when calibrating gates with a target
                angle per gate of :math:`\pi/2` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is False by default when calibrating gates with
                a target angle per gate of :math:`\pi/2`.
            repetitions (list[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
            target_angle (float): The target angle per gate.
        """
        options = super()._default_experiment_options()
        options.add_cal_circuits = False
        options.repetitions = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        options.target_angle = np.pi / 2
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            'sx12',
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            gate=SX12Gate()
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": np.pi,
            }
        )

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        repetitions = np.array(self.experiment_options.repetitions)
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = 0.5 * np.cos((np.pi / 2. + 0.01) * repetitions) + 0.5
        num_qubits = 1

        if self.experiment_options.add_cal_circuits:
            one_probs = np.concatenate(([1., 0.], one_probs))

        if self.run_options.meas_level == MeasLevel.KERNELED:
            if self._final_xgate:
                states = (0, 2)
            else:
                states = (1, 2)

            meas_return=self.run_options.get('meas_return', 'avg')

            return ef_memory(one_probs, shots, num_qubits, meas_return,
                             states=states)
        return single_qubit_counts(one_probs, shots, num_qubits)
