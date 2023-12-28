"""Override of qiskit_experiments.framework.composite.batch_experiment."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.result import Counts
from qiskit_experiments.framework import (BaseExperiment, BatchExperiment as BatchExperimentOrig,
                                          ParallelExperiment as ParallelExperimentOrig)

from .composite_analysis import CompositeAnalysis

if TYPE_CHECKING:
    from qiskit.providers import Backend


class BatchExperiment(BatchExperimentOrig):
    """BatchExperiment with modified functionalities.

    Modifications:
    - Use the overridden CompositeAnalysis by default.
    - Add an optional physical_qubits argument to the constructor.
    - Bug fix for _remap_qubits.
    - Add dummy_data().
    """
    def __init__(
        self,
        experiments: list[BaseExperiment],
        physical_qubits: Optional[Sequence[int]] = None,
        backend: Optional['Backend'] = None,
        flatten_results: bool = False,
        analysis: Optional[CompositeAnalysis] = None,
    ):
        if analysis is None:
            analysis = CompositeAnalysis(
                [exp.analysis for exp in experiments], flatten_results=flatten_results
            )

        super().__init__(experiments, backend=backend, flatten_results=flatten_results,
                         analysis=analysis)

        if physical_qubits is not None:
            # We'll have to overwrite the following three attributes
            #  self._num_qubits = 1 (set in BaseExperiment)
            #  self._physical_qubits = [physical_qubits[1]] (set in BaseExperiment)
            #  self._qubit_map = {physical_qubits[1]: 0} (set in BatchExperiment)
            if not set(physical_qubits) >= set(self._physical_qubits):
                raise QiskitError('Provided physical_qubits is not a superset of component'
                                  ' experiment physical_qubits.')

            self._num_qubits = len(physical_qubits)
            self._physical_qubits = tuple(physical_qubits)
            if self._num_qubits != len(set(self._physical_qubits)):
                raise QiskitError("Duplicate qubits in physical qubits list.")

            self._qubit_map = {ph: lo for lo, ph in enumerate(physical_qubits)}

    def _remap_qubits(self, circuit, qubit_mapping):
        """Bug fix (new_circuit.append does not preserve calibrations)"""
        num_qubits = self.num_qubits
        num_clbits = circuit.num_clbits
        new_circuit = QuantumCircuit(num_qubits, num_clbits, name="batch_" + circuit.name)
        new_circuit.metadata = circuit.metadata
        new_circuit.compose(circuit, qubits=qubit_mapping, inplace=True)
        return new_circuit

    def dummy_data(
        self,
        transpiled_circuits: list[QuantumCircuit]
    ) -> list[Union[np.ndarray, Counts]]:
        """Generate dummy data from component experiments."""
        data = []

        cstart = 0
        for exp in self._experiments:
            if isinstance(exp, ParallelExperimentOrig):
                num_circuits = len(exp.component_experiment(0).circuits())
            else:
                num_circuits = len(exp.circuits())

            circuits = [c.copy() for c in transpiled_circuits[cstart:cstart + num_circuits]]
            # Flatten the metadata
            for circuit in circuits:
                composite_metadata = circuit.metadata.pop('composite_metadata')[0]
                circuit.metadata.update(composite_metadata)

            data.extend(exp.dummy_data(circuits))
            cstart += num_circuits

        return data
