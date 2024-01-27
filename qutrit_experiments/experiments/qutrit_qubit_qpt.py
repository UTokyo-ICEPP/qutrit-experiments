from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework.compound_analysis import CompoundAnalysis
from ..gates import X12Gate
from ..transpilation import map_to_physical_qubits
from ..util.bloch import paulis, rotation_matrix_xyz
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
        options.plot = True
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
        fidelities = []

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            fit_res = child_data.analysis_results('unitary_fit_result').value
            unitary_parameters.append(np.array(fit_res.params))
            fidelities.append(1. - fit_res.state.error)

        analysis_results.extend([
            AnalysisResultData(name='unitary_parameters', value=np.array(unitary_parameters)),
            AnalysisResultData(name='fidelities', value=np.array(fidelities))
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='Pauli expectation'
            )
            data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])
            initial_states = np.array([
                [1., 0.],
                [0., 1.],
                [1. / np.sqrt(2.), 1. / np.sqrt(2.)],
                [1. / np.sqrt(2.), 1.j / np.sqrt(2.)]
            ])
            # Z, X, Y
            meas_bases = paulis[[2, 0, 1]]

            y_preds = []
            for control_state in range(3):
                child_data = experiment_data.child_data(component_index[control_state])
                yval = data_processor(child_data.data())
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=np.arange(12),
                    y_formatted=unp.nominal_values(yval),
                    y_formatted_err=unp.std_devs(yval)
                )
                fit_res = child_data.analysis_results('unitary_fit_result').value
                unitary = rotation_matrix_xyz(fit_res.params)
                evolved = np.einsum('ij,sj->si', unitary, initial_states)
                y_preds.append(np.einsum('si,mik,sk->sm',
                                         evolved.conjugate(), meas_bases, evolved).real)

            figure = plotter.figure()
            ax = figure.axes[0]
            # assuming Pauli bases
            # Zp, Zm, Xp, Yp
            for control_state, y_pred in enumerate(y_preds):
                ax.bar(np.arange(12), np.zeros(12), 1., bottom=y_pred.reshape(-1), fill=False,
                       edgecolor=plotter.drawer.DefaultColors[control_state],
                       label=f'c{control_state} fit')
            ax.set_ylim(-1.05, 1.05)
            ax.legend()

            figures.append(figure)

        return analysis_results, figures
