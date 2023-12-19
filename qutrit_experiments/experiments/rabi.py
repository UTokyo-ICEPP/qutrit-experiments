"""Rabi experiment with optimized transpilation."""
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_experiments.library import Rabi as RabiOrig

from ..transpilation import replace_calibration_and_metadata

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
