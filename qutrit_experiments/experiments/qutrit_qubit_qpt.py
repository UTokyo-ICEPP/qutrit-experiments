from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework.compound_analysis import CompoundAnalysis
from ..gates import X12Gate
from ..transpilation import map_to_physical_qubits
from .process_tomography import CircuitTomography


class QutritQubitQPT(BatchExperiment):
    """Target-qubit QPT for three initial states of the control qubit."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        experiments = []
        for control_state in range(3):
            channel = QuantumCircuit(2)
            if control_state >= 1:
                channel.x(0)
            if control_state == 2:
                channel.append(X12Gate(), [0])
            channel.barrier()
            channel.compose(circuit, inplace=True)

            experiments.append(
                CircuitTomography(channel, physical_qubits=physical_qubits,
                                  measurement_indices=[1], preparation_indices=[1], backend=backend,
                                  extra_metadata={'control_state': control_state})
            )

        super().__init__(experiments, backend=backend,
                         analysis=QutritQubitQPTAnalysis([exp.analysis for exp in experiments]))
        self.extra_metadata = extra_metadata

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        if self.extra_metadata:
            metadata.update(self.extra_metadata)
        return metadata

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        prep_circuits = self._experiments[0]._decomposed_prep_circuits()
        meas_circuits = self._experiments[0]._decomposed_meas_circuits()
        batch_circuits = []
        for index, exp in enumerate(self._experiments):
            channel = exp._circuit
            if to_transpile:
                channel = map_to_physical_qubits(channel, self.physical_qubits,
                                                 self._backend.coupling_map)
            expr_circuits = exp._compose_qpt_circuits(channel, prep_circuits, meas_circuits,
                                                      to_transpile)
            for circuit in expr_circuits:
                # Update metadata
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [circuit.metadata],
                    "composite_index": [index],
                }
                batch_circuits.append(circuit)

        return batch_circuits


class QutritQubitQPTAnalysis(CompoundAnalysis):
    """Analysis for QutritQubitQPT."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = None # Needed to have DP propagated to QPT analysis
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        """Compute the rotation parameters θ that maximize the fidelity between exp(-i/2 θ.σ) and
        the observed Choi matrices."""
        component_index = experiment_data.metadata['component_child_index']

        unitary_parameters = []

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            fit_res = child_data.analysis_results('unitary_fit_result').value
            unitary_parameters.append(np.array(fit_res.params))

        analysis_results.append(
            AnalysisResultData(name='unitary_parameters', value=np.array(unitary_parameters))
        )

        return analysis_results, figures
