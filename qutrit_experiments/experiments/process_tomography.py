"""QPT experiment with custom target circuit."""
from collections.abc import Iterable, Sequence
import logging
from typing import Any, Optional
from matplotlib.figure import Figure
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.result import Counts, CorrelatedReadoutMitigator, LocalReadoutMitigator
from qiskit_experiments.data_processing import DataProcessor, Probability, BasisExpectationValue
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.library.tomography import basis, ProcessTomographyAnalysis
from qiskit_experiments.library.tomography.fitters.cvxpy_utils import cvxpy
from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment

from ..constants import DEFAULT_SHOTS
from ..data_processing import ReadoutMitigation
from ..framework.threaded_analysis import NO_THREAD, ThreadedAnalysis
from ..transpilation import map_to_physical_qubits, map_and_translate, translate_to_basis
from ..util.unitary_fit import fit_unitary, plot_unitary_fit

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


class CircuitTomography(TomographyExperiment):
    """QPT experiment with custom target circuit."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.decompose_circuits = False
        options.need_translation = False
        options.pre_circuit = None
        options.post_circuit = None
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
        if target_circuit is not None:
            target = Operator(target_circuit)
        else:
            target = None

        analysis = CircuitTomographyAnalysis()
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

        self.extra_metadata = extra_metadata or {}

    def circuits(self) -> list[QuantumCircuit]:
        if self.experiment_options.decompose_circuits:
            return self._decomposed_circuits(apply_layout=False)

        circuits = super().circuits()
        for circuit in circuits:
            for inst in list(circuit.data):
                if inst.operation.name == 'reset':
                    circuit.data.remove(inst)

        if (pre_circuit := self.experiment_options.pre_circuit) is not None:
            new_circuits = []
            for original in circuits:
                circuit = original.copy_empty_like()
                circuit.compose(pre_circuit, inplace=True)
                circuit.compose(original, inplace=True)
                new_circuits.append(circuit)
            circuits = new_circuits

        if (post_circuit := self.experiment_options.post_circuit) is not None:
            for circuit in circuits:
                circuit.compose(post_circuit, inplace=True)

        if len(self._prep_physical_qubits) * len(self._meas_physical_qubits) == 1:
            self._add_basis_metadata(circuits)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return self._decomposed_circuits(apply_layout=True)

    def _decomposed_circuits(self, apply_layout=False) -> list[QuantumCircuit]:
        channel = self._circuit
        if apply_layout:
            if self.experiment_options.need_translation:
                channel = map_and_translate(channel, self.physical_qubits, self._backend)
            else:
                channel = map_to_physical_qubits(channel, self.physical_qubits,
                                                 self._backend.coupling_map)

        prep_circuits = self._decomposed_prep_circuits()
        meas_circuits = self._decomposed_meas_circuits()
        return self._compose_qpt_circuits(channel, prep_circuits, meas_circuits, apply_layout)

    def _decomposed_prep_circuits(self) -> list[QuantumCircuit]:
        pbasis = self._prep_circ_basis
        pqubits = self._prep_physical_qubits
        prep_shape = pbasis.index_shape(pqubits)
        prep_circuits = []
        for physical_qubit, basis_size in zip(pqubits, prep_shape):
            qubit_prep_circuits = [pbasis.circuit((idx,), [physical_qubit]).decompose()
                                   for idx in range(basis_size)]
            prep_circuits.append(
                translate_to_basis(qubit_prep_circuits, self._backend)
            )
        return prep_circuits

    def _decomposed_meas_circuits(self) -> list[QuantumCircuit]:
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
        pqubits = self._prep_physical_qubits
        mqubits = self._meas_physical_qubits
        circuits = super().circuits()
        pre_circuit = self.experiment_options.pre_circuit
        post_circuit = self.experiment_options.post_circuit
        qpt_circuits = []
        for circuit, (prep_element, meas_element) in zip(circuits, self._basis_indices()):
            qpt_circuit = channel.copy_empty_like(name=circuit.name)
            qpt_circuit.metadata = circuit.metadata
            qpt_circuit.add_register(ClassicalRegister(len(mqubits)))

            if pre_circuit:
                qpt_circuit.compose(pre_circuit, inplace=True)

            for iq, idx in enumerate(prep_element):
                if apply_layout:
                    qubit = pqubits[iq]
                else:
                    qubit = self._prep_indices[iq]
                qpt_circuit.compose(prep_circuits[iq][idx], qubits=[qubit], inplace=True)

            qpt_circuit.barrier()
            qpt_circuit.compose(channel, inplace=True)
            qpt_circuit.barrier()

            for iq, idx in enumerate(meas_element):
                if apply_layout:
                    qubit = mqubits[iq]
                else:
                    qubit = self._meas_indices[iq]

                qpt_circuit.compose(meas_circuits[iq][idx], qubits=[qubit], inplace=True)

            if post_circuit:
                qpt_circuit.compose(post_circuit, inplace=True)

            qpt_circuit.barrier()
            if apply_layout:
                qpt_circuit.measure(mqubits, range(len(mqubits)))
            else:
                qpt_circuit.measure(self._meas_indices, range(len(mqubits)))

            qpt_circuits.append(qpt_circuit)

        if len(pqubits) * len(mqubits) == 1:
            self._add_basis_metadata(qpt_circuits)

        return qpt_circuits

    def _add_basis_metadata(self, circuits: list[QuantumCircuit]):
        # Assuming Pauli basis
        prep_states = ['z', 'z', 'x', 'y']
        prep_signs = [1, -1, 1, 1]
        meas_bases = ['z', 'x', 'y']
        for circuit, (prep_element, meas_element) in zip(circuits, self._basis_indices()):
            circuit.metadata['initial_state'] = prep_states[prep_element[0]]
            circuit.metadata['initial_state_sign'] = prep_signs[prep_element[0]]
            circuit.metadata['meas_basis'] = meas_bases[meas_element[0]]

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
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


class CircuitTomographyAnalysis(ThreadedAnalysis, ProcessTomographyAnalysis):
    """ProcessTomographyAnalysis with an optional fit to a unitary when number of qubits is 1."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.readout_mitigator = None
        options.plot = True
        return options

    def _run_analysis_threaded(self, experiment_data: ExperimentData) -> Any:
        num_qubits = len(experiment_data.metadata['m_qubits'])
        num_qubits *= len(experiment_data.metadata['p_qubits'])
        if num_qubits == 1:
            if self.options.readout_mitigator is not None:
                nodes = [
                    ReadoutMitigation(readout_mitigator=self.options.readout_mitigator,
                                      physical_qubits=self.physical_qubits),
                    Probability('1'),
                    BasisExpectationValue()
                ]
                data_processor = DataProcessor('counts', nodes)
            else:
                data_processor = None

            return fit_unitary(experiment_data.data(), data_processor)
        else:
            return None

    def _run_analysis_unthreaded(
        self,
        experiment_data: ExperimentData,
        thread_output: Any
    ) -> tuple[list[AnalysisResultData, list[Figure]]]:
        if isinstance(self.options.readout_mitigator, LocalReadoutMitigator):
            readout_mitigator = self.options.readout_mitigator
        elif isinstance(self.options.readout_mitigator, CorrelatedReadoutMitigator):
            matrices = [self.options.readout_mitigator.assignment_matrix(qubit)
                        for qubit in self.physical_qubits]
            readout_mitigator = LocalReadoutMitigator(matrices, self.physical_qubits)

            logger.debug('Overwriting counts using the output of the DataProcessor.')
            processed_ydata = self.options.data_processor(experiment_data.data())
            original_counts = [datum['counts'] for datum in experiment_data.data()]
            for datum, ydatum in zip(experiment_data.data(), processed_ydata):
                datum['counts'] = ydatum
        else:
            readout_mitigator = None

        mmt_basis = self.options.measurement_basis
        if readout_mitigator is not None:
            self.set_options(
                measurement_basis=basis.PauliMeasurementBasis(mitigator=readout_mitigator)
            )

        analysis_results, figures = super(ThreadedAnalysis, self)._run_analysis(experiment_data)

        self.set_options(measurement_basis=mmt_basis)

        if thread_output:
            popt_ufloats, state, expvals_pred, fit_input = thread_output
            analysis_results.extend([
                AnalysisResultData(name='unitary_fit_state', value=state),
                AnalysisResultData(name='unitary_fit_params', value=popt_ufloats),
                AnalysisResultData(name='expvals_observed', value=fit_input[0]),
                AnalysisResultData(name='expvals_predicted', value=expvals_pred)
            ])
            if self.options.plot:
                figures.append(plot_unitary_fit(*fit_input))

        return analysis_results, figures
