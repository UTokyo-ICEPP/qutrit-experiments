"""QPT experiment with custom target circuit."""
from collections.abc import Iterable, Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.result import Counts
from qiskit_experiments.library.tomography import basis, ProcessTomographyAnalysis
from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment
from qiskit_experiments.library.tomography.fitters.cvxpy_utils import cvxpy

from ..constants import DEFAULT_SHOTS
from ..transpilation.layout_only import map_to_physical_qubits
from ..transpilation.layout_and_translation import map_and_translate


class CircuitTomography(TomographyExperiment):
    """QPT experiment with custom target circuit."""
    def __init__(
        self,
        circuit: QuantumCircuit,
        target_circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_indices: Optional[Sequence[int]] = None,
        preparation_basis: basis.PreparationBasis = basis.PauliPreparationBasis(),
        preparation_indices: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[tuple[list[int], list[int]]]] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        target = Operator(target_circuit)

        analysis = ProcessTomographyAnalysis()
        analysis.set_options(target=target)
        if cvxpy is not None:
            analysis.set_options(fitter='cvxpy_gaussian_lstsq')

        super().__init__(
            circuit,
            backend=backend,
            physical_qubits=physical_qubits,
            measurement_basis=measurement_basis,
            measurement_indices=measurement_indices,
            preparation_basis=preparation_basis,
            preparation_indices=preparation_indices,
            basis_indices=basis_indices,
            analysis=analysis
        )

        self.extra_metadata = extra_metadata

    def circuits(self) -> list[QuantumCircuit]:
        circs = super().circuits()
        for circuit in circs:
            for inst in list(circuit.data):
                if inst.operation.name == 'reset':
                    circuit.data.remove(inst)

            circuit.metadata['qubits'] = self.physical_qubits

        return circs

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        pbasis = self._prep_circ_basis
        pqubits = self._prep_physical_qubits
        mbasis = self._meas_circ_basis
        mqubits = self._meas_physical_qubits

        prep_shape = pbasis.index_shape(pqubits)
        meas_shape = mbasis.index_shape(mqubits)

        channel = map_to_physical_qubits(self._circuit.decompose(), self.physical_qubits,
                                         self.transpile_options.target)

        prep_circuits = []
        for physical_qubit, basis_size in zip(pqubits, prep_shape):
            prep_circuits.append([
                map_and_translate(pbasis.circuit((idx,), [physical_qubit]).decompose(),
                                  [physical_qubit], self.transpile_options.target)
                for idx in range(basis_size)
            ])

        meas_circuits = []
        for physical_qubit, basis_size in zip(mqubits, meas_shape):
            circuits = [
                map_and_translate(mbasis.circuit((idx,), [physical_qubit]).decompose(),
                                  [physical_qubit], self.transpile_options.target)
                for idx in range(basis_size)
            ]
            for circuit in circuits:
                # All single-qubit circuits -> always uses creg0
                circuit.remove_final_measurements()
            meas_circuits.append(circuits)

        circuits = self.circuits()
        transpiled_circuits = []

        for circuit, (prep_element, meas_element) in zip(circuits, self._basis_indices()):
            transpiled = channel.copy_empty_like(name=circuit.name)
            transpiled.metadata = circuit.metadata
            transpiled.add_register(ClassicalRegister(len(mqubits)))

            for iq, idx in enumerate(prep_element):
                transpiled.compose(prep_circuits[iq][idx], inplace=True)
            transpiled.barrier()
            transpiled.compose(channel, inplace=True)
            transpiled.barrier()
            for iq, idx in enumerate(meas_element):
                transpiled.compose(meas_circuits[iq][idx], inplace=True)
            transpiled.barrier()
            transpiled.measure(mqubits, range(len(mqubits)))

            transpiled_circuits.append(transpiled)

        return transpiled_circuits

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        if self.extra_metadata is not None:
            metadata.update(self.extra_metadata)

        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        shots = self.run_options.get('shots', DEFAULT_SHOTS)

        data = []

        rng = np.random.default_rng()

        nq = len(self.physical_qubits)

        init = np.array([1.] + [0.] * (2 ** nq - 1))

        for prep_element, meas_element in self._basis_indices():
            prep_circ = self._prep_circ_basis.circuit(prep_element, self._prep_physical_qubits)
            meas_circ = self._meas_circ_basis.circuit(meas_element, self._meas_physical_qubits)
            meas_circ.remove_final_measurements(inplace=True)

            op = Operator(prep_circ) & self.analysis.options.target & Operator(meas_circ)
            statevector = op.to_matrix() @ init
            probs = np.square(np.abs(statevector))

            icounts = rng.multinomial(shots, probs)

            counts = Counts(dict(enumerate(icounts)), memory_slots=nq)

            data.append(counts)

        return data
