"""QPT experiment with custom target circuit."""
from collections.abc import Iterable, Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.result import Counts
from qiskit_experiments.framework import Options
from qiskit_experiments.library.tomography import basis, ProcessTomographyAnalysis
from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment
from qiskit_experiments.library.tomography.fitters.cvxpy_utils import cvxpy

from ..constants import DEFAULT_SHOTS
from ..transpilation import map_and_translate, map_to_physical_qubits, translate_to_basis


class CircuitTomography(TomographyExperiment):
    """QPT experiment with custom target circuit."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.decompose_circuits = False
        return options

    def __init__(
        self,
        circuit: QuantumCircuit,
        target_circuit: Optional[QuantumCircuit] = None,
        backend: Optional[Backend] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        measurement_basis: basis.MeasurementBasis = basis.PauliMeasurementBasis(),
        measurement_indices: Optional[Sequence[int]] = None,
        preparation_basis: basis.PreparationBasis = basis.PauliPreparationBasis(),
        preparation_indices: Optional[Sequence[int]] = None,
        basis_indices: Optional[Iterable[tuple[list[int], list[int]]]] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        if target_circuit:
            target = Operator(target_circuit)
        else:
            target = None

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
        if self.experiment_options.decompose_circuits:
            return self._decomposed_circuits(apply_layout=False)
        
        circs = super().circuits()
        for circuit in circs:
            for inst in list(circuit.data):
                if inst.operation.name == 'reset':
                    circuit.data.remove(inst)

        return circs

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return self._decomposed_circuits(apply_layout=True)
    
    def _decomposed_circuits(self, apply_layout=False) -> list[QuantumCircuit]:
        channel = self._circuit
        if apply_layout:
            channel = map_to_physical_qubits(channel, self.physical_qubits,
                                             self._backend.coupling_map)

        prep_circuits = self._decomposed_prep_circuits(apply_layout)
        meas_circuits = self._decomposed_meas_circuits(apply_layout)
        return self._compose_qpt_circuits(channel, prep_circuits, meas_circuits, apply_layout)

    
    def _decomposed_prep_circuits(self, apply_layout: bool) -> list[QuantumCircuit]:
        pbasis = self._prep_circ_basis
        pqubits = self._prep_physical_qubits
        prep_shape = pbasis.index_shape(pqubits)
        prep_circuits = []
        for physical_qubit, basis_size in zip(pqubits, prep_shape):
            qubit_prep_circuits = [pbasis.circuit((idx,), [physical_qubit]).decompose()
                                   for idx in range(basis_size)]
            if apply_layout:
                prep_circuits.append(
                    map_and_translate(qubit_prep_circuits, [physical_qubit], self._backend)
                )
            else:
                prep_circuits.append(
                    translate_to_basis(qubit_prep_circuits, self._backend)
                )
        return prep_circuits

    def _decomposed_meas_circuits(self, apply_layout: bool) -> list[QuantumCircuit]:
        mbasis = self._meas_circ_basis
        mqubits = self._meas_physical_qubits
        meas_shape = mbasis.index_shape(mqubits)
        meas_circuits = []
        for physical_qubit, basis_size in zip(mqubits, meas_shape):
            qubit_meas_circuits = [mbasis.circuit((idx,), [physical_qubit]).decompose()
                                   for idx in range(basis_size)]
            for circuit in qubit_meas_circuits:
                # All single-qubit circuits -> always uses creg0
                circuit.remove_final_measurements()

            if apply_layout:
                meas_circuits.append(
                    map_and_translate(qubit_meas_circuits, [physical_qubit], self._backend)
                )
            else:
                meas_circuits.append(
                    translate_to_basis(qubit_meas_circuits, self._backend)
                )
        return meas_circuits
    
    def _compose_qpt_circuits(
        self,
        channel: QuantumCircuit,
        prep_circuits: list[list[QuantumCircuit]],
        meas_circuits: list[list[QuantumCircuit]],
        apply_layout: bool
    ) -> list[QuantumCircuit]:
        mqubits = self._meas_physical_qubits
        circuits = super().circuits()
        decomposed_circuits = []
        for circuit, (prep_element, meas_element) in zip(circuits, self._basis_indices()):
            decomposed = channel.copy_empty_like(name=circuit.name)
            decomposed.metadata = circuit.metadata
            decomposed.add_register(ClassicalRegister(len(mqubits)))

            for iq, idx in enumerate(prep_element):
                decomposed.compose(prep_circuits[iq][idx], inplace=True)
            decomposed.barrier()
            decomposed.compose(channel, inplace=True)
            decomposed.barrier()
            for iq, idx in enumerate(meas_element):
                decomposed.compose(meas_circuits[iq][idx], inplace=True)
            decomposed.barrier()
            if apply_layout:
                decomposed.measure(mqubits, range(len(mqubits)))
            else:
                decomposed.measure(self._meas_indices, range(len(mqubits)))

            decomposed_circuits.append(decomposed)
        return decomposed_circuits

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        if self.extra_metadata is not None:
            metadata.update(self.extra_metadata)

        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]: # pylint: disable=unused-argument
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
