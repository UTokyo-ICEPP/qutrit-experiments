from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData)
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ..experiment_mixins import MapToPhysicalQubits


class TruthTable(MapToPhysicalQubits, BaseExperiment):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, backend=backend)
        self.set_experiment_options(circuit=circuit)
        self.analysis = TruthTableAnalysis()

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []
        num_qubits = len(self._physical_qubits)
        for init in range(2 ** num_qubits):
            bstr = ('{:0%db}' % num_qubits).format(init)
            bits = [iq for iq in range(num_qubits) if bstr[num_qubits - iq - 1] == '1']
            circuit = QuantumCircuit(num_qubits)
            if bits:
                circuit.x(bits)
            circuit.compose(self.experiment_options.circuit, inplace=True)
            circuit.measure_all()
            circuit.metadata = {'init': init}
            circuits.append(circuit)

        return circuits


class TruthTableAnalysis(BaseAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = DataProcessor('counts', [])
        options.plot = True
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        num_qubits = len(experiment_data.metadata['physical_qubits'])
        num_states = 2 ** num_qubits
        truth_table = np.zeros((num_states, num_states))

        counts_arr = self.options.data_processor(experiment_data.data())

        for counts, datum in zip(counts_arr, experiment_data.data()):
            init = datum['metadata']['init']
            shots = datum['shots']
            for bstr, count in counts.items():
                truth_table[init, int(bstr, 2)] = count / shots

        analysis_results = [AnalysisResultData(name='truth_table', value=truth_table)]

        if self.options.plot:
            ax = get_non_gui_ax()
            ax.matshow(truth_table, cmap=plt.cm.binary, clim=[0, 1]) # pylint: disable=no-member
            ax.set_xlabel("Initial state")
            ax.xaxis.set_label_position("top")
            ax.set_ylabel("Measured State")
            ax.set_xticks(np.arange(num_states), labels=[f'{i}' for i in range(num_states)])
            ax.set_yticks(np.arange(num_states), labels=[f'{i}' for i in range(num_states)])
            return analysis_results, [ax.get_figure()]
        return analysis_results, []
