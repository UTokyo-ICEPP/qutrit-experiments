from collections.abc import Sequence
from typing import Any, Optional
import jax
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import XGate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.framework import AnalysisResultData, ExperimentData

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
            qpt_circuit = QuantumCircuit(2, 1)
            if control_state >= 1:
                qpt_circuit.x(0)
            if control_state == 2:
                qpt_circuit.append(X12Gate(), [0])
            qpt_circuit.barrier()
            qpt_circuit.compose(circuit, inplace=True)

            experiments.append(
                CircuitTomography(qpt_circuit, physical_qubits=physical_qubits,
                                  measurement_indices=[1], preparation_indices=[1],backend=backend,
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
        prep_circuits = self._experiments[0]._decomposed_prep_circuits(to_transpile)
        meas_circuits = self._experiments[0]._decomposed_meas_circuits(to_transpile)
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
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        return [], []
        component_index = experiment_data.metadata['component_child_index']

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            choi = child_data.analysis_results('state')[0].value

        return experiment_data, figures
