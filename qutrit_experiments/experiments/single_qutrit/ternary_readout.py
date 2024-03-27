"""Ternary readout error confusion measurement."""
from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ...experiment_mixins import MapToPhysicalQubits
from ...gates import X12Gate


class MCMLocalReadoutError(MapToPhysicalQubits, BaseExperiment):
    """Ternary readout error confusion measurement using mid-circuit measurements."""
    @classmethod
    def _default_run_options(cls) -> Options:
        options = super()._default_run_options()
        options.rep_delay = 5.e-4
        return options
    
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
        meas_template = QuantumCircuit(1, 2)
        meas_template.measure(0, 0)
        meas_template.x(0)
        meas_template.measure(0, 1)

        circuits = []
        for state, gates in enumerate([(), (XGate(),), (XGate(), X12Gate())]):
            circ = QuantumCircuit(1, 2)
            for gate in gates:
                circ.append(gate, [0])
            circ.compose(meas_template, inplace=True)
            circ.metadata = {
                'experiment_type': self._type,
                'qubit': self.physical_qubits[0],
                'state_label': state
            }
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
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        assignment_matrix = np.zeros((4, 4))

        for datum in experiment_data.data():
            in_state = datum['metadata']['state_label']
            for obs_state, key in enumerate(['10', '01', '11', '00']):
                assignment_matrix[obs_state, in_state] = datum['counts'].get(key, 0)

            assignment_matrix[:, in_state] /= np.sum(assignment_matrix[:, in_state])

        analysis_results = [AnalysisResultData(name='assignment_matrix', value=assignment_matrix)]

        if self.options.plot:
            ax = get_non_gui_ax()
            ax.matshow(assignment_matrix, cmap=plt.cm.binary, clim=[0, 1]) # pylint: disable=no-member
            ax.set_xlabel("Prepared State")
            ax.xaxis.set_label_position("top")
            ax.set_ylabel("Measured State")
            ax.set_xticks(np.arange(4), labels=['0', '1', '2', ''])
            ax.set_yticks(np.arange(4), labels=['0', '1', '2', 'invalid'])
            return analysis_results, [ax.get_figure()]

        return analysis_results, []
