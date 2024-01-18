"""Rabi experiment with optimized transpilation."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_experiments.framework import Options
from qiskit_experiments.library import Rabi as RabiOrig

from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..gates import QutritGate
from ..transpilation import map_to_physical_qubits
from ..util.dummy_data import from_one_probs


class Rabi(RabiOrig):
    """Rabi experiment with optimized transpilation."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                          self._backend_data.coupling_map,
                                          common_circuit_optimization=True)
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

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[np.ndarray]: # pylint: disable=unused-argument
        amplitudes = self.experiment_options.amplitudes
        one_probs = np.cos(2. * np.pi * 4. * amplitudes) * 0.49 + 0.51
        return from_one_probs(self, one_probs)
