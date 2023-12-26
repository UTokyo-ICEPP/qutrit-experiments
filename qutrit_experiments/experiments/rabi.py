"""Rabi experiment with optimized transpilation."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options
from qiskit_experiments.library import Rabi as RabiOrig

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..gates import QutritGate
from ..transpilation import replace_calibration_and_metadata
from ..util.dummy_data import ef_memory, single_qubit_counts


class Rabi(RabiOrig):
    """Rabi experiment with optimized transpilation."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                    self._backend_data.coupling_map)
        # Need to update the gate parameters too
        # CircuitInstruction.operation.params is a list of ParameterExpressions
        amp = Parameter('amp')
        for circuit in circuits[1:]:
            # Amplitude is rounded in RabiOrig
            param_value = list(circuit.calibrations[self.__gate_name__].keys())[0][1][0]
            rabi = next(inst for inst in circuit.data if inst.operation.name == self.__gate_name__)
            rabi.operation.params = [amp.assign(amp, param_value)]

        return circuits


class EFRabi(EFSpaceExperiment, Rabi):
    """Rabi experiment with initial and final X gates."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(-0.4, 0.4, 17)
        return options

    def circuits(self) -> list[QuantumCircuit]:
        circuits = super().circuits()

        for circuit in circuits:
            for inst in circuit.data:
                if (op := inst.operation).name == self.__gate_name__:
                    inst.operation = QutritGate(op.name, 1, list(op.params))

        return circuits

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]:
        amplitudes = self.experiment_options.amplitudes
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = np.cos(2. * np.pi * 4. * amplitudes) * 0.49 + 0.51
        num_qubits = 1

        if self.experiment_options.discrimination_basis == '01':
            states = (0, 1)
        elif self.experiment_options.discrimination_basis == '02':
            states = (0, 2)
        else:
            states = (1, 2)

        if self.run_options.meas_level == MeasLevel.KERNELED:
            return ef_memory(one_probs, shots, num_qubits,
                             meas_return=self.run_options.get('meas_return', 'avg'),
                             states=states)

        return single_qubit_counts(one_probs, shots, num_qubits)
