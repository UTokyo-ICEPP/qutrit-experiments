from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.providers import Backend, Options
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework_overrides.composite_analysis import CompositeAnalysis
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
        expval_obss = []
        expval_preds = []

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            fit_state = child_data.analysis_results('unitary_fit_state').value
            popt = child_data.analysis_results('unitary_fit_params').value
            unitary_parameters.append(popt)
            fidelities.append(1. - fit_state.error)
            expval_obss.append(child_data.analysis_results('expvals_observed').value)
            expval_preds.append(child_data.analysis_results('expvals_predicted').value)

        analysis_results.extend([
            AnalysisResultData(name='unitary_parameters', value=np.array(unitary_parameters)),
            AnalysisResultData(name='fidelities', value=np.array(fidelities)),
            AnalysisResultData(name='expvals_observed', value=np.array(expval_obss)),
            AnalysisResultData(name='expvals_predicted', value=np.array(expval_preds))
        ])

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='Pauli expectation'
            )

            for control_state in range(3):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=np.arange(12),
                    y_formatted=unp.nominal_values(expval_obss[control_state]),
                    y_formatted_err=unp.std_devs(expval_obss[control_state])
                )

            figure = plotter.figure()
            ax = figure.axes[0]
            # assuming Pauli bases
            # Zp, Zm, Xp, Yp
            for control_state, expval_pred in enumerate(expval_preds):
                ax.bar(np.arange(12), np.zeros(12), 1., bottom=expval_pred.reshape(-1), fill=False,
                       edgecolor=plotter.drawer.DefaultColors[control_state],
                       label=f'c{control_state} fit')
            ax.set_ylim(-1.05, 1.05)
            ax.legend()

            figures.append(figure)

        return analysis_results, figures


class QutritQubitQPTScan(BatchExperiment):
    """BatchExperiment of QutritQubitQPT scanning over variables."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.template_circuit = None
        options.parameters = None
        options.parameter_values = None
        options.angle_parameters_map = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        param_name: Union[str, Sequence[str]],
        values: Union[Sequence[float], Sequence[Sequence[float]]],
        angle_param_name: Optional[Union[str, dict[str, str]]] = None,
        backend: Optional[Backend] = None,
        analysis: Optional[CompositeAnalysis] = None
    ):
        def find_param(pname, plist):
            return next(p for p in plist if p.name == pname)

        if isinstance(param_name, str):
            param_name = [param_name]
        params = [find_param(pname, circuit.parameters) for pname in param_name]

        try:
            len(values[0])
        except TypeError:
            values = [values]

        assert len(set(len(v) for v in values)) == 1, 'Values list must have a common length'

        if angle_param_name is None:
            angle_param_name = {}
        elif isinstance(angle_param_name, str):
            angle_param_name = {param_name[0]: angle_param_name}
        angle_params = {find_param(key, params): find_param(value, circuit.parameters)
                        for key, value in angle_param_name.items()}

        experiments = []
        for exp_values in zip(*values):
            assign_params = self._make_assign_map(exp_values, params, angle_params)
            experiments.append(
                QutritQubitQPT(physical_qubits,
                               circuit.assign_parameters(assign_params, inplace=False),
                               backend=backend,
                               extra_metadata={p.name: v for p, v in zip(params, exp_values)})
            )

        if analysis is None:
            analysis = QutritQubitQPTScanAnalysis([exp.analysis for exp in experiments])
        super().__init__(experiments, backend=backend, analysis=analysis)

        self.set_experiment_options(
            template_circuit=circuit,
            parameters=params,
            parameter_values=values,
            angle_parameters_map=angle_params
        )

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['scan_parameters'] = [p.name for p in self.experiment_options.parameters]
        metadata['scan_values'] = [list(v) for v in self.experiment_options.parameter_values]
        return metadata

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        dummy_experiment = QutritQubitQPT(self.physical_qubits,
                                          self.experiment_options.template_circuit,
                                          backend=self._backend)
        try:
            # Can we transpile without assigning values?
            template_circuits = dummy_experiment._batch_circuits(to_transpile)
        except:
            # If not, just revert to BatchExperiment default behavior
            return super()._batch_circuits(to_transpile=to_transpile)

        circuits = []
        for index, exp_values in enumerate(zip(*self.experiment_options.parameter_values)):
            assign_params = self._make_assign_map(exp_values)
            for template_circuit in template_circuits:
                circuit = template_circuit.assign_parameters(assign_params, inplace=False)
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [template_circuit.metadata],
                    "composite_index": [index],
                }
                circuits.append(circuit)
        return circuits

    def _make_assign_map(self, exp_values: tuple[float, ...], params=None, angle_params=None):
        if params is None:
            params = self.experiment_options.parameters
        if angle_params is None:
            angle_params = self.experiment_options.angle_parameters_map
        assign_params = dict(zip(params, exp_values))
        for param, angle_param in angle_params.items():
            if assign_params[param] < 0.:
                assign_params[param] *= -1.
                assign_params[angle_param] = np.pi
            else:
                assign_params[angle_param] = 0.
        return assign_params


class QutritQubitQPTScanAnalysis(CompoundAnalysis):
    """Analysis for QutritQubitQPTScan."""
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
        component_index = experiment_data.metadata['component_child_index']
        parameters = experiment_data.metadata['scan_parameters']
        scan_values = [np.array(v) for v in experiment_data.metadata['scan_values']]

        unitaries = []
        fidelities = []
        expval_obss = []
        expval_preds = []
        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            unitaries.append(child_data.analysis_results('unitary_parameters').value)
            fidelities.append(child_data.analysis_results('fidelities').value)
            expval_obss.append(child_data.analysis_results('expvals_observed').value)
            expval_preds.append(child_data.analysis_results('expvals_predicted').value)

        unitaries = np.array(unitaries)
        fidelities = np.array(fidelities)
        expval_obss = np.array(expval_obss)
        expval_preds = np.array(expval_preds)

        analysis_results.extend([
            AnalysisResultData(name=p, value=v) for p, v in zip(parameters, scan_values)
        ])
        analysis_results.extend([
            AnalysisResultData(name='unitary_parameters', value=unitaries),
            AnalysisResultData(name='fidelities', value=fidelities),
            AnalysisResultData(name='expvals_observed', value=expval_obss),
            AnalysisResultData(name='expvals_predicted', value=expval_preds)
        ])

        if self.options.plot:
            for iop, op in enumerate(['X', 'Y', 'Z']):
                plotter = CurvePlotter(MplDrawer())
                plotter.set_figure_options(
                    xlabel=parameters[0],
                    ylabel=f'Unitary {op} parameter',
                    ylim=(-np.pi - 0.1, np.pi + 0.1)
                )
                for control_state in range(3):
                    plotter.set_series_data(
                        f'c{control_state}',
                        x_formatted=scan_values[0],
                        y_formatted=unitaries[:, control_state, iop],
                        y_formatted_err=np.zeros_like(scan_values[0])
                    )
                figures.append(plotter.figure())

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='QPT fidelity',
                ylim=(-0.1, 1.1)
            )
            for control_state in range(3):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=np.arange(len(scan_values[0])),
                    y_formatted=fidelities[:, control_state],
                    y_formatted_err=np.zeros_like(scan_values[0])
                )
            figures.append(plotter.figure())

            chisq = np.sum(
                np.square((unp.nominal_values(expval_obss) - expval_preds)
                           / unp.std_devs(expval_obss)),
                axis=2
            )
            chisq /= expval_obss.shape[2]

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='chisq'
            )
            for control_state in range(3):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=np.arange(len(scan_values[0])),
                    y_formatted=chisq[:, control_state],
                    y_formatted_err=np.zeros_like(scan_values[0])
                )
            figures.append(plotter.figure())

        return analysis_results, figures