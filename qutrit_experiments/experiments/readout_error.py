"""Readout confusion matrix measurements."""
from collections.abc import Sequence
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.library import CorrelatedReadoutError as CorrelatedReadoutErrorOrig

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins import MapToPhysicalQubits
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


class MCMLocalReadoutError(MapToPhysicalQubits, BaseExperiment):
    """Ternary readout error confusion measurement using mid-circuit measurements."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=MCMLocalReadoutErrorAnalysis(),
                         backend=backend)

    def circuits(self) -> list[QuantumCircuit]:
        template = QuantumCircuit(1, 2)
        template.metadata = {
            "experiment_type": self._type,
            "qubit": self.physical_qubits[0],
        }

        circuits = []

        circ = template.copy()
        circ.measure(0, 0)
        circ.barrier(0)
        circ.x(0)
        circ.barrier(0)
        circ.measure(0, 1)
        circ.metadata['state_label'] = 0
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.barrier(0)
        circ.measure(0, 0)
        circ.barrier(0)
        circ.x(0)
        circ.barrier(0)
        circ.measure(0, 1)
        circ.metadata['state_label'] = 1
        circuits.append(circ)

        circ = template.copy()
        circ.x(0)
        circ.append(X12Gate(), [0])
        circ.barrier(0)
        circ.measure(0, 0)
        circ.barrier(0)
        circ.x(0)
        circ.barrier(0)
        circ.measure(0, 1)
        circ.append(X12Gate(), [0]) # To allow active reset
        circ.metadata['state_label'] = 2
        circuits.append(circ)

        return circuits


class MCMLocalReadoutErrorAnalysis(BaseAnalysis):
    """Analysis for MCMLocalReadoutError."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        assignment_matrix = np.zeros((4, 4))

        for datum in experiment_data.data():
            in_state = datum['metadata']['state_label']
            for obs_state, key in enumerate(['10', '01', '11', '00']):
                assignment_matrix[obs_state, in_state] = datum['counts'].get(key, 0)

            assignment_matrix[:, in_state] /= np.sum(assignment_matrix[:, in_state])

        analysis_results = [AnalysisResultData(name='assignment_matrix', value=assignment_matrix)]

        if self.options.plot:
            ax = get_non_gui_ax()
            ax.matshow(assignment_matrix, cmap=plt.cm.binary, clim=[0, 1])
            ax.set_xlabel("Prepared State")
            ax.xaxis.set_label_position("top")
            ax.set_ylabel("Measured State")
            ax.set_xticks(np.arange(4), labels=['0', '1', '2', ''])
            ax.set_yticks(np.arange(4), labels=['0', '1', '2', 'invalid'])
            return analysis_results, [ax.get_figure()]

        return analysis_results, []
