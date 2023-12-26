"""Readout confusion matrix measurements."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction, ClassicalRegister, Measure
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.library import CorrelatedReadoutError as CorrelatedReadoutErrorOrig

from ..constants import DEFAULT_SHOTS
from ..gates import X12Gate
from ..transpilation.layout_only import map_to_physical_qubits


class CorrelatedReadoutError(CorrelatedReadoutErrorOrig):
    """Override of CorrelatedReadoutError with custom transpilation."""
    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.shots = 10000
        return options

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = self.circuits()
        first_circuit = map_to_physical_qubits(circuits[0], self.physical_qubits,
                                               self._backend.coupling_map)

        transpiled_circuits = [first_circuit]
        # first_circuit is for 0
        for circuit in circuits[1:]:
            tcirc = first_circuit.copy()
            state_label = circuit.metadata['state_label']
            for i, val in enumerate(reversed(state_label)):
                if val == "1":
                    tcirc.data.insert(0, CircuitInstruction(XGate(), [self.physical_qubits[i]]))

            tcirc.metadata = circuit.metadata
            transpiled_circuits.append(tcirc)

        return transpiled_circuits

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        template = '{:0%db}' % self.num_qubits
        return [Counts({template.format(state): shots}) for state in range(2 ** self.num_qubits)]


class MCMLocalReadoutError(BaseExperiment):
    """Ternary readout error confusion measurement using mid-circuit measurements."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=MCMLocalReadoutErrorAnalysis(),
                         backend=backend)

    def circuits(self) -> list[QuantumCircuit]:
        template = QuantumCircuit(1, 1)
        template.metadata = {
            "experiment_type": self._type,
            "qubit": self.physical_qubits[0],
        }

        circuits = []

        circ = template.copy()
        circ.measure(0, 0)
        circ.metadata['state_label'] = '0'
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.measure(0, 0)
        circ.metadata['state_label'] = '1'
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.append(X12Gate(), [0])
        circ.measure(0, 0)
        circ.append(X12Gate(), [0]) # To allow active reset
        circ.metadata['state_label'] = '2'
        circuits.append(circ)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = map_to_physical_qubits(self.circuits(), self.physical_qubits,
                                          self._backend.coupling_map)

        for circ in circuits:
            creg = ClassicalRegister(size=2)
            circ.add_register(creg)
            measure_idx, inst_orig = next((item for item in enumerate(circ.data)
                                           if isinstance(item[1].operation, Measure)))
            # Insert measure-X-measure in reverse order
            # Post-transpilation - logical qubit = physical qubit
            inst_orig.operation = Measure()
            inst_orig.clbits = (creg[1],)
            circ.data.insert(measure_idx,
                             CircuitInstruction(XGate(), qubits=inst_orig.qubits))
            circ.data.insert(measure_idx,
                             CircuitInstruction(Measure(), qubits=inst_orig.qubits,
                                                clbits=(creg[0],)))

        return circuits


class MCMLocalReadoutErrorAnalysis(BaseAnalysis):
    """Analysis for MCMLocalReadoutError."""
    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        assignment_matrix = np.zeros(4, 4)

        for datum in experiment_data.data():
            in_state = int(datum['metadata']['state_label'])
            for obs_state, key in enumerate(['10', '01', '11', '00']):
                assignment_matrix[obs_state, in_state] = datum['counts'].get(key, 0)

            assignment_matrix[:, in_state] /= np.sum(assignment_matrix[:, in_state])

        return [AnalysisResultData(name='assignment_matrix', value=assignment_matrix)], []
