"""Override of qiskit_experiments.framework.composite.parallel_experiment."""

from collections import defaultdict
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.result import Counts
from qiskit_experiments.framework import (BaseExperiment,
                                          ParallelExperiment as ParallelExperimentOrig)

from .composite_analysis import CompositeAnalysis


class ParallelExperiment(ParallelExperimentOrig):
    """ParallelExperiment with modified functionalities.

    Modifications:
    - Use the overridden CompositeAnalysis by default.
    - Add dummy_data().
    """
    def __init__(
        self,
        experiments: list[BaseExperiment],
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

    def dummy_data(
        self,
        transpiled_circuits: list[QuantumCircuit]
    ) -> list[Union[np.ndarray, Counts]]:
        """Generate dummy data from component experiments."""
        transpiled_sub_circuits = []

        for circuit in transpiled_circuits:
            # THIS IS ACTUALLY WRONG - cargs MUST BE REMAPPED
            sub_circuits = []
            sub_circuit_map = {}
            for qubits in circuit.metadata['composite_qubits']:
                for qubit in qubits:
                    sub_circuit_map[qubit] = len(sub_circuits)

                sub_circuits.append(circuit.copy_empty_like())

            for inst, qargs, cargs in circuit.data:
                icirc = sub_circuit_map[circuit.find_bit(qargs[0]).index]
                sub_circuits[icirc]._append(inst, qargs, cargs)

            transpiled_sub_circuits.append(sub_circuits)

        # Transpose the list structure to [experiment, circuit]
        transpiled_sub_circuits = list(zip(*transpiled_sub_circuits))

        sub_dummy_data = [exp.dummy_data(circuits)
                          for exp, circuits in zip(self._experiments, transpiled_sub_circuits)]

        # Re-transpose to [circuit, experiment]
        sub_dummy_data = list(zip(*sub_dummy_data))

        if self.run_options.meas_level == MeasLevel.KERNELED:
            meas_return = self.run_options.get('meas_return', MeasReturnType.SINGLE)
            if meas_return == MeasReturnType.SINGLE:
                axis = 1
            else:
                axis = 0

            data = [np.concatenate(dummy_data, axis=axis) for dummy_data in sub_dummy_data]
        else:
            data = []

            for dummy_data in sub_dummy_data:
                # List out the outcomes per experiment per shot to make concatenation easier
                outcomes = []
                for counts in dummy_data:
                    outcomes.append(sum(([key] * cnt for key, cnt in counts.items()), []))

                # Invert the order so that experiment 0 goes to the right
                outcomes = list(reversed(outcomes))
                # Transpose -> [shot, experiment]
                outcomes = list(zip(*outcomes))
                # Concatenate -> [shot]
                outcomes = [''.join(exp_outcome) for exp_outcome in outcomes]

                counts = defaultdict(int)
                for outcome in outcomes:
                    counts[outcome] += 1

                data.append(Counts(counts))

        return data
