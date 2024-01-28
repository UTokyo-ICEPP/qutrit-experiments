"""QPT experiment with custom target circuit."""
from collections.abc import Iterable, Sequence
import logging
from typing import Any, Optional
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.providers import Backend
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.channel.transformations import _to_superop
from qiskit.result import Counts
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.library.tomography import basis, ProcessTomographyAnalysis
from qiskit_experiments.library.tomography.fitters.cvxpy_utils import cvxpy
from qiskit_experiments.library.tomography.tomography_experiment import TomographyExperiment
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..constants import DEFAULT_SHOTS
from ..transpilation import map_and_translate, map_to_physical_qubits, translate_to_basis
from ..util.bloch import paulis, rotation_matrix_xyz

logger = logging.getLogger(__name__)
twopi = 2. * np.pi


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
        qpt_circuits = []
        for circuit, (prep_element, meas_element) in zip(circuits, self._basis_indices()):
            qpt_circuit = channel.copy_empty_like(name=circuit.name)
            qpt_circuit.metadata = circuit.metadata
            qpt_circuit.add_register(ClassicalRegister(len(mqubits)))

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

            qpt_circuit.barrier()
            if apply_layout:
                qpt_circuit.measure(mqubits, range(len(mqubits)))
            else:
                qpt_circuit.measure(self._meas_indices, range(len(mqubits)))

            qpt_circuits.append(qpt_circuit)
        return qpt_circuits

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


class CircuitTomographyAnalysis(ProcessTomographyAnalysis):
    """ProcessTomographyAnalysis with an optional fit to a unitary when number of qubits is 1."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.unitary_parameters_p0 = None
        options.data_processor = None
        options.plot = True
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData, list['matplotlib.figure.Figure']]]:
        if self.options.data_processor:
            original_counts = self._update_data(experiment_data)

        analysis_results, figures = super()._run_analysis(experiment_data)

        num_qubits = len(experiment_data.metadata['m_qubits'])
        num_qubits *= len(experiment_data.metadata['p_qubits'])
        if num_qubits != 1:
            if self.options.data_processor:
                self._restore_data(experiment_data, original_counts)
            return analysis_results, figures

        choi = next(res for res in analysis_results if res.name == 'state').value
        channel = _to_superop("Choi", choi, 2, 2)

        @jax.jit
        def infidelity(params):
            unitary = jax.lax.cond(
                jnp.allclose(params, 0.),
                lambda params: jnp.eye(2, dtype='complex128'),
                lambda params: rotation_matrix_xyz(params, npmod=jnp),
                params
            )
            target = jnp.kron(jnp.conj(unitary), unitary)
            return 1. - jnp.trace(target.T.conjugate() @ channel).real / 4

        solver = jaxopt.GradientDescent(fun=infidelity)

        if (init := self.options.unitary_parameters_p0) is None:
            init = np.array([np.pi, 0., 0.])

        res = solver.run(init)
        params = np.array(res.params)
        while (pnorm := np.sqrt(np.sum(np.square(params)))) > twopi:
            params *= (1. - twopi / pnorm)
        if (pnorm := np.sqrt(np.sum(np.square(params)))) > np.pi:
            params *= -(twopi - pnorm) / pnorm

        data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
        expval_obs = data_processor(experiment_data.data())

        # assuming Pauli bases
        # Zp, Zm, Xp, Yp
        initial_states = np.array([
            [1., 0.],
            [0., 1.],
            [1. / np.sqrt(2.), 1. / np.sqrt(2.)],
            [1. / np.sqrt(2.), 1.j / np.sqrt(2.)]
        ])
        # Z, X, Y
        meas_bases = paulis[[2, 0, 1]]
        unitary = rotation_matrix_xyz(params)
        evolved = np.einsum('ij,sj->si', unitary, initial_states)
        expval_pred = np.einsum('si,mik,sk->sm', evolved.conjugate(), meas_bases, evolved).real
        expval_pred = expval_pred.reshape(-1)

        analysis_results.extend([
            AnalysisResultData(name='unitary_fit_state', value=res.state),
            AnalysisResultData(name='unitary_fit_params', value=params),
            AnalysisResultData(name='expvals_observed', value=expval_obs),
            AnalysisResultData(name='expvals_predicted', value=expval_pred)
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='Pauli expectation'
            )

            plotter.set_series_data(
                'qpt',
                x_formatted=np.arange(12),
                y_formatted=unp.nominal_values(expval_obs),
                y_formatted_err=unp.std_devs(expval_obs)
            )
            figure = plotter.figure()
            ax = figure.axes[0]
            ax.bar(np.arange(12), np.zeros(12), 1., bottom=expval_pred.reshape(-1), fill=False,
                   label='fit result')
            ax.set_ylim(-1.05, 1.05)
            ax.legend()

            figures.append(figure)

        if self.options.data_processor:
            self._restore_data(experiment_data, original_counts)
        return analysis_results, figures

    def _update_data(self, experiment_data: ExperimentData):
        logger.debug('Overwriting counts using the output of the DataProcessor.')
        processed_ydata = self.options.data_processor(experiment_data.data())
        original_counts = [datum['counts'] for datum in experiment_data.data()]
        for datum, ydatum in zip(experiment_data.data(), processed_ydata):
            datum['counts'] = ydatum
        return original_counts

    def _restore_data(self, experiment_data: ExperimentData, original_counts: list[dict[str, int]]):
        logger.debug('Restoring counts.')
        for datum, counts in zip(experiment_data.data(), original_counts):
            datum['counts'] = counts
