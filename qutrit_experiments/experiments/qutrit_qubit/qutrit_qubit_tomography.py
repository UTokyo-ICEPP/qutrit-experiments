"""Qubit process/unitary tomography with control qutrit in three states."""
from collections.abc import Sequence
import logging
from threading import Lock
from typing import Any, Optional, Union
import jax
import jax.numpy as jnp
import jaxopt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from uncertainties import correlated_values, unumpy as unp
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit_experiments.data_processing import (BasisExpectationValue, DataProcessor,
                                                MarginalizeCounts, Probability)
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...data_processing import MultiProbability, ReadoutMitigation
from ...framework_overrides.batch_experiment import BatchExperiment
from ...framework_overrides.composite_analysis import CompositeAnalysis
from ...framework.combined_analysis import CombinedAnalysis
from ...gates import X12Gate
from ...transpilation import map_to_physical_qubits
from ...util.bloch import so3_cartesian, so3_cartesian_params
from ..process_tomography import CircuitTomography
from ..unitary_tomography import UnitaryTomography

logger = logging.getLogger(__name__)


class QutritQubitTomography(BatchExperiment):
    """Target-qubit Tomography for three initial states of the control qubit.

    To properly characterize the input circuit, tomography must be performed on the combination
    of initial state preparation plus the circuit. We therefore perform tomographies of initial
    state preparation circuits independently to cancel its effect.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: Union[QuantumCircuit, Gate],
        tomography_type: str = 'unitary',
        measure_preparations: bool = True,
        measure_qutrit: bool = False,
        control_states: Sequence[int] = (0, 1, 2),
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        analysis_cls: Optional[type[CombinedAnalysis]] = None
    ):
        if isinstance(circuit, Gate):
            gate = circuit
            circuit = QuantumCircuit(2)
            circuit.append(gate, [0, 1])

        experiments = []
        for iexp in range(5 if measure_preparations else 3):
            control_state = iexp if iexp < 3 else iexp - 2
            if control_state not in control_states:
                continue

            post_circuit = QuantumCircuit(2)
            if measure_qutrit:
                post_circuit.add_register(ClassicalRegister(3))
                post_circuit.measure(0, 1)
                post_circuit.x(0)
                post_circuit.append(X12Gate(), [0])
                post_circuit.measure(0, 2)

            channel = QuantumCircuit(2)
            if control_state >= 1:
                channel.x(0)
            if control_state == 2:
                channel.append(X12Gate(), [0])

            if iexp < 3:
                channel.compose(circuit, inplace=True)

            if tomography_type == 'unitary':
                exp = UnitaryTomography(physical_qubits, channel, backend=backend,
                                        extra_metadata={'control_state': control_state})
                exp.set_experiment_options(
                    measured_logical_qubit=1,
                    post_circuit=post_circuit
                )
            else:
                exp = CircuitTomography(channel, physical_qubits=physical_qubits,
                                        measurement_indices=[1], preparation_indices=[1],
                                        backend=backend,
                                        extra_metadata={'control_state': control_state})
                exp.set_experiment_options(
                    post_circuit=post_circuit
                )
            if iexp >= 3:
                exp.extra_metadata['state_preparation'] = True

            experiments.append(exp)

        if analysis_cls is None:
            analysis_cls = QutritQubitTomographyAnalysis

        super().__init__(experiments, backend=backend,
                         analysis=analysis_cls([exp.analysis for exp in experiments]))
        self.tomography_type = tomography_type
        self.extra_metadata = extra_metadata or {}
        if measure_qutrit:
            nodes = [MarginalizeCounts({0}), Probability('1'), BasisExpectationValue()]
            # DataProcessor is propagated to subexperiment analyses automatically
            self.analysis.set_options(
                analyze_qutrit=True,
                data_processor=DataProcessor('counts', nodes)
            )

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)
        metadata['control_states'] = sorted(set(exp.extra_metadata['control_state']
                                                for exp in self._experiments))
        return metadata

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        if self.tomography_type == 'unitary':
            return super()._batch_circuits(to_transpile)
        else:
            prep_circuits = self._experiments[0]._decomposed_prep_circuits()
            meas_circuits = self._experiments[0]._decomposed_meas_circuits()
            batch_circuits = []
            for index, exp in enumerate(self._experiments):
                channel = exp._circuit
                if to_transpile:
                    channel = map_to_physical_qubits(channel, self.physical_qubits,
                                                     self._backend.coupling_map)
                expr_circuits = exp._compose_tomography_circuits(channel, prep_circuits,
                                                                 meas_circuits, to_transpile)
                for circuit in expr_circuits:
                    # Update metadata
                    circuit.metadata = {
                        "composite_metadata": [circuit.metadata],
                        "composite_index": [index],
                    }
                    batch_circuits.append(circuit)

            return batch_circuits


class QutritQubitTomographyAnalysis(CombinedAnalysis):
    """Analysis for QutritQubitTomography."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names = ['expectation_values']
        options.data_processor = None  # Needed to have DP propagated to tomography analysis
        options.plot = True
        options.parallelize = 0  # This analysis is somehow faster done in serial
        options.prep_unitaries = {}
        options.maxiter = None
        options.tol = None
        options.analyze_qutrit = False
        options.qutrit_assignment_matrix = None
        return options

    @classmethod
    def _propagated_option_keys(cls) -> list[str]:
        return super()._propagated_option_keys() + ['maxiter', 'tol']

    def _run_combined_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata['component_child_index']

        unitary_parameters = {}
        prep_unitary_parameters = {}
        observeds = {}
        predicteds = {}

        for iexp in range(len(self._analyses)):
            child_data = experiment_data.child_data(component_index[iexp])
            control_state = child_data.metadata['control_state']
            popt = child_data.analysis_results('unitary_fit_params').value
            if child_data.metadata.get('state_preparation', False):
                prep_unitary_parameters[control_state] = popt
                key = (control_state, 'prep')
            else:
                unitary_parameters[control_state] = popt
                key = (control_state, '')
            observeds[key] = child_data.analysis_results('expvals_observed').value
            predicteds[key] = child_data.analysis_results('expvals_predicted').value

        if not prep_unitary_parameters:
            prep_unitary_parameters = self.options.prep_unitaries
        if prep_unitary_parameters:
            analysis_results.extend([
                AnalysisResultData(name='raw_parameters', value=dict(unitary_parameters)),
                AnalysisResultData(name='prep_parameters', value=dict(prep_unitary_parameters))
            ])
            for control_state, prep_params in prep_unitary_parameters.items():
                unitary = (so3_cartesian(unitary_parameters[control_state], npmod=unp)
                           @ so3_cartesian(-prep_params, npmod=unp))
                unitary_parameters[control_state] = so3_cartesian_params(unitary, npmod=unp)

        analysis_results.extend([
            AnalysisResultData(name='unitary_parameters', value=unitary_parameters),
            AnalysisResultData(name='expvals_observed', value=observeds),
            AnalysisResultData(name='expvals_predicted', value=predicteds)
        ])

        if self.options.analyze_qutrit:
            nodes = [MarginalizeCounts({1, 2})]
            if (amat := self.options.qutrit_assignment_matrix) is not None:
                nodes.append(ReadoutMitigation(amat))
            nodes.append(MultiProbability(['10', '01', '11', '00'], [0.5] * 4))
            data_processor = DataProcessor('counts', nodes)

            qutrit_states = {}
            for iexp in range(len(self._analyses)):
                child_data = experiment_data.child_data(component_index[iexp])
                control_state = child_data.metadata['control_state']
                if child_data.metadata.get('state_preparation', False):
                    key = (control_state, 'prep')
                else:
                    key = (control_state, '')
                qutrit_data = data_processor(child_data.data())
                qutrit_states[key] = np.array(
                    [[datum[outcome] for outcome in ['10', '01', '11', '00']]
                     for datum in qutrit_data]
                )

            analysis_results.append(
                AnalysisResultData(name='qutrit_states', value=qutrit_states)
            )

        if self.options.plot:
            # Plot all expectation values
            labels = []
            child_data = experiment_data.child_data(component_index[0])
            for datum in child_data.data():
                metadata = datum['metadata']
                label = metadata['meas_basis']
                label += '|' + metadata['initial_state']
                label += '+' if metadata.get('initial_state_sign', 1) > 0 else '-'
                labels.append(label)
            ax = get_non_gui_ax()
            xvalues = np.arange(list(observeds.values())[0].shape[0])
            ax.set_xticks(xvalues, labels=labels)
            ax.set_ylim(-1.05, 1.05)
            ax.set_ylabel('Pauli expectation')
            ax.axhline(0., color='black', linestyle='--', linewidth=0.1)
            for key in sorted(observeds.keys()):
                label = f'c={key[0]}'
                if key[1]:
                    label += ' prep'
                ec = ax.errorbar(xvalues, unp.nominal_values(observeds[key]),
                                 unp.std_devs(observeds[key]), fmt='o',
                                 label=label)
                ax.bar(xvalues, np.zeros_like(xvalues), 1., bottom=predicteds[key],
                       fill=False, edgecolor=ec.lines[0].get_markerfacecolor())

            ax.legend()
            figures.append(ax.get_figure())

            if self.options.analyze_qutrit:
                ax = get_non_gui_ax()
                keys = sorted(qutrit_states.keys(), key=lambda x: (x[1], x[0]))
                xvalues = (np.arange(len(keys)) + 0.5) * list(qutrit_states.values())[0].shape[0]
                ax.set_xticks(xvalues, labels=[f'{key[0]} {key[1]}' for key in keys])
                yvalues = np.arange(4)
                ax.set_yticks(yvalues, labels=['0', '1', '2', 'invalid'])
                ax.set_ylim(-0.5, 3.5)
                ax.set_ylabel('Final state')
                for istate, key in enumerate(keys):
                    svalues = unp.nominal_values(qutrit_states[key])
                    nx = svalues.shape[0]
                    xvalues = np.linspace(nx * istate, nx * (istate + 1), nx, endpoint=False)
                    xvalues += 0.5
                    xvalues = np.repeat(xvalues, 4)
                    yvalues = np.tile(np.arange(4), svalues.shape[0])
                    ax.errorbar(xvalues, yvalues, yerr=svalues.reshape(-1) * 0.5, fmt='none')

                figures.append(ax.get_figure())
                if len(self.options.figure_names) == 1:
                    self.options.figure_names.append('qutrit_states')

        return analysis_results, figures


class QutritQubitTomographyScan(BatchExperiment):
    """BatchExperiment of QutritQubitTomography scanning over variables."""
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
        circuit: Union[QuantumCircuit, Gate],
        param_name: Union[str, Sequence[str]],
        values: Union[Sequence[float], Sequence[Sequence[float]]],
        angle_param_name: Optional[Union[str, dict[str, str]]] = None,
        tomography_type: str = 'unitary',
        measure_preparations: bool = True,
        control_states: Sequence[int] = (0, 1, 2),
        backend: Optional[Backend] = None,
        analysis_cls: Optional[type[CompositeAnalysis]] = None
    ):
        if isinstance(circuit, Gate):
            gate = circuit
            circuit = QuantumCircuit(2)
            circuit.append(gate, [0, 1])

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
        for iexp, exp_values in enumerate(zip(*values)):
            assign_params = self._make_assign_map(exp_values, params, angle_params)
            extra_metadata = {p.name: v for p, v in zip(params, exp_values)}
            experiments.append(
                QutritQubitTomography(physical_qubits,
                                      circuit.assign_parameters(assign_params, inplace=False),
                                      tomography_type=tomography_type,
                                      measure_preparations=(measure_preparations and iexp == 0),
                                      control_states=control_states, backend=backend,
                                      extra_metadata=extra_metadata)
            )

        if analysis_cls is None:
            analysis_cls = QutritQubitTomographyScanAnalysis

        super().__init__(experiments, backend=backend,
                         analysis=analysis_cls([exp.analysis for exp in experiments]))
        self.tomography_type = tomography_type
        self.control_states = tuple(control_states)

        self.set_experiment_options(
            template_circuit=circuit,
            parameters=params,
            parameter_values=values,
            angle_parameters_map=angle_params
        )

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['control_states'] = list(self.control_states)
        metadata['scan_parameters'] = [p.name for p in self.experiment_options.parameters]
        metadata['scan_values'] = [list(v) for v in self.experiment_options.parameter_values]
        return metadata

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        dummy_exp = QutritQubitTomography(self.physical_qubits,
                                          self.experiment_options.template_circuit,
                                          tomography_type=self.tomography_type,
                                          measure_preparations=True,
                                          control_states=self.control_states,
                                          backend=self._backend)
        try:
            # Can we transpile without assigning values?
            template_circuits = dummy_exp._batch_circuits(to_transpile)
        except Exception:  # pylint: disable=broad-exception-caught
            # If not, just revert to BatchExperiment default behavior
            return super()._batch_circuits(to_transpile=to_transpile)

        num_tomography_circuits = (len(self.control_states)
                                   * len(dummy_exp.component_experiment(0).circuits()))
        circuits = []
        for iexp, exp_values in enumerate(zip(*self.experiment_options.parameter_values)):
            assign_params = self._make_assign_map(exp_values)
            for template_circuit in template_circuits[:num_tomography_circuits]:
                circuit = template_circuit.assign_parameters(assign_params, inplace=False)
                circuit.metadata = {
                    "composite_metadata": [template_circuit.metadata],
                    "composite_index": [iexp],
                }
                circuits.append(circuit)

            if self.component_experiment(iexp).num_experiments > len(self.control_states):
                for template_circuit in template_circuits[num_tomography_circuits:]:
                    circuit = template_circuit.copy()
                    circuit.metadata = {
                        "composite_metadata": [template_circuit.metadata],
                        "composite_index": [iexp],
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


class QutritQubitTomographyScanAnalysis(CombinedAnalysis):
    """Analysis for QutritQubitTomographyScan."""
    def __init_subclass__(cls, **kwargs):
        # Each subclass needs its own lock and cache
        cls._compile_lock = Lock()
        cls._fit_functions_cache = {}
        return super().__init_subclass__(**kwargs)

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names = ['theta_x', 'theta_y', 'theta_z', 'chisq']
        options.data_processor = None  # Needed to have DP propagated to tomography analysis
        options.parallelize_on_thread = False
        options.simul_fit = False
        options.plot = True
        options.parallelize = 0  # This analysis is somehow faster done in serial
        options.return_expvals = False
        options.maxiter = None
        options.tol = None
        options.unitary_parameter_ylims = {}
        return options

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        if self.options.simul_fit:
            self.options.parallelize_on_thread = True

        for an in self._analyses:
            if (val := self.options.maxiter):
                an.set_options(maxiter=val)
            if (val := self.options.tol):
                an.set_options(tol=val)

    def _run_combined_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata['component_child_index']
        control_states = experiment_data.metadata['control_states']
        # Get the prep unitary params from the first child data
        first_child = experiment_data.child_data(component_index[0])
        try:
            prep_params = first_child.analysis_results('prep_parameters').value
            prep_inverses = {state: so3_cartesian(-params, npmod=unp)
                             for state, params in prep_params.items()}
        except ExperimentEntryNotFound:
            prep_params = {}
            prep_inverses = None

        # Compile the child data fit results into a dict keyed on control state
        # Optionally correct for preparation
        unitaries = {state: [] for state in control_states}
        observeds = {state: [] for state in control_states}
        predicteds = {state: [] for state in control_states}
        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            unitary_params = child_data.analysis_results('unitary_parameters').value
            observed = child_data.analysis_results('expvals_observed').value
            predicted = child_data.analysis_results('expvals_predicted').value
            if prep_inverses is not None and child_index != component_index[0]:
                # first child data is already corrected
                for state, cancellation in prep_inverses.items():
                    unitary = so3_cartesian(unitary_params[state], npmod=unp) @ cancellation
                    unitary_params[state] = so3_cartesian_params(unitary, npmod=unp)

            for state in control_states:
                unitaries[state].append(unitary_params[state])
                observeds[state].append(observed[(state, '')])
                predicteds[state].append(predicted[(state, '')])

        for lists in [unitaries, observeds, predicteds]:
            for state in control_states:
                lists[state] = np.array(lists[state])

        chisq = {
            state: np.mean(
                np.square(
                    (unp.nominal_values(observeds[state]) - predicteds[state])
                    / unp.std_devs(observeds[state])
                ),
                axis=1
            ) for state in control_states
        }

        analysis_results.extend([
            AnalysisResultData(name='control_states', value=control_states),
            AnalysisResultData(name='unitary_parameters', value=unitaries),
            AnalysisResultData(name='chisq', value=chisq)
        ])
        if self.options.return_expvals:
            analysis_results.extend([
                AnalysisResultData(name='expvals_observed', value=observeds),
                AnalysisResultData(name='expvals_predicted', value=predicteds)
            ])

        if self.options.simul_fit:
            # Array of init and meas should have common shapes for all xvals
            # In fact the shape shouldn't depend on control state either unless we do something
            # really strange
            initial_states = {state: [] for state in control_states}
            meas_bases = {state: [] for state in control_states}
            axes = ['x', 'y', 'z']
            for ut_index in first_child.metadata['component_child_index']:
                ut_data = first_child.child_data(ut_index)
                if ut_data.metadata.get('state_preparation', False):
                    continue
                control_state = ut_data.metadata['control_state']
                for datum in ut_data.data():
                    metadata = datum['metadata']
                    initial_states[control_state].append(axes.index(metadata['initial_state']))
                    meas_bases[control_state].append(axes.index(metadata['meas_basis']))

            for lists in [initial_states, meas_bases]:
                for state in control_states:
                    lists[state] = np.array(lists[state])

            # The scanned parameter must be the first in scan_values list
            xvals = np.array(experiment_data.metadata['scan_values'][0])
            popt_ufloats = {}
            for state in control_states:
                if (params := prep_params.get(state)) is not None:
                    prep = so3_cartesian(unp.nominal_values(params))
                else:
                    prep = np.eye(3)

                logger.debug('Performing simultaneous fit for control state %d', state)
                popt_ufloats[state] = self.simultaneous_fit(
                    xvals, initial_states[state], meas_bases[state], observeds[state], prep,
                    unitaries[state]
                )

            analysis_results.append(
                AnalysisResultData('simul_fit_params', value=popt_ufloats)
            )

        if self.options.plot:
            parameters = experiment_data.metadata['scan_parameters']
            scan_values = experiment_data.metadata['scan_values']

            plotters = []
            for iop, op in enumerate(['X', 'Y', 'Z']):
                plotter = CurvePlotter(MplDrawer())
                plotter.set_figure_options(
                    xlabel=parameters[0],
                    ylabel=f'Unitary {op} parameter'
                )
                if (ylim := self.options.unitary_parameter_ylims.get(op)):
                    plotter.set_figure_options(ylim=ylim)
                else:
                    plotter.set_figure_options(ylim=(-np.pi - 0.1, np.pi + 0.1))
                for control_state in control_states:
                    plotter.set_series_data(
                        f'c{control_state}',
                        x_formatted=scan_values[0],
                        y_formatted=unp.nominal_values(unitaries[control_state][:, iop]),
                        y_formatted_err=unp.std_devs(unitaries[control_state][:, iop])
                    )
                plotters.append(plotter)

            if self.options.simul_fit:
                x_interp = np.linspace(scan_values[0][0], scan_values[0][-1], 100)
                xyz_preds = {state: self.unitary_params(unp.nominal_values(popt_ufloats[state]),
                                                        x_interp)
                             for state in control_states}

                for iop, plotter in enumerate(plotters):
                    for control_state in control_states:
                        plotter.set_series_data(
                            f'c{control_state}',
                            x_interp=x_interp,
                            y_interp=xyz_preds[control_state][:, iop]
                        )

            figures.extend([plotter.figure() for plotter in plotters])

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='chisq'
            )
            for state in control_states:
                plotter.set_series_data(
                    f'c{state}',
                    x_formatted=np.arange(len(scan_values[0])),
                    y_formatted=chisq[state],
                    y_formatted_err=np.zeros_like(scan_values[0])
                )

                if self.options.simul_fit:
                    unitary_params = self.unitary_params(
                        unp.nominal_values(popt_ufloats[state]), scan_values[0]
                    )
                    bloch_coords = so3_cartesian(unitary_params)
                    preds = bloch_coords[:, meas_bases[state], initial_states[state]]
                    y_interp = np.mean(
                        np.square(
                            (unp.nominal_values(observeds[state]) - preds)
                            / unp.std_devs(observeds[state])
                        ),
                        axis=1
                    )
                    plotter.set_series_data(
                        f'c{state}',
                        x_interp=np.arange(len(scan_values[0])),
                        y_interp=y_interp
                    )

            figures.append(plotter.figure())

        return analysis_results, figures

    def simultaneous_fit(
        self,
        xvals: np.ndarray,
        initial_states: np.ndarray,
        meas_bases: np.ndarray,
        expvals: np.ndarray,
        prep_unitary: np.ndarray,
        unitary_params: np.ndarray
    ):
        logger.debug('Simul fit input:\nxvals %s\nexpvals %s\ninitial_states %s\nmeas_bases %s',
                     xvals, expvals, initial_states, meas_bases)
        # Use normalized xvals throughout
        xvals_norm = xvals[-1] - xvals[0]
        norm_xvals = xvals / xvals_norm

        expvals_n = unp.nominal_values(expvals)
        expvals_e = unp.std_devs(expvals)
        expvals_e_norm = expvals_e / np.mean(expvals_e)
        fit_args = (norm_xvals, meas_bases, initial_states, expvals_n, expvals_e_norm, prep_unitary)
        p0s = self._get_p0s(norm_xvals, xvals_norm, unitary_params)
        logger.debug('Initial parameters %s', p0s)
        vobj, vsolve, hess = self.fit_functions(p0s.shape, expvals.shape)

        fit_result = vsolve(p0s, *fit_args)
        fvals = vobj(fit_result.params, *fit_args)
        iopt = np.argmin(fvals)
        popt = np.array(fit_result.params[iopt])
        logger.debug('Optimal parameters %s', popt)

        hess_args = (norm_xvals, meas_bases, initial_states, expvals_n, expvals_e, prep_unitary)
        pcov = np.linalg.inv(hess(popt, *hess_args))
        upopt = np.array(correlated_values(popt, pcov))
        self._postprocess_params(upopt, xvals_norm)
        logger.debug('Adjusted parameters %s', upopt)
        return upopt

    def _get_p0s(self, norm_xvals: np.ndarray, xvals_norm: float, unitary_params: np.ndarray):
        raise NotImplementedError()

    # pylint: disable-next=unused-argument
    def _postprocess_params(self, upopt: np.ndarray, norm: float):
        return

    @classmethod
    def fit_functions(cls, params_shape: tuple[int, ...], expvals_shape: tuple[int, int]):
        key = (params_shape, expvals_shape)
        with cls._compile_lock:
            if (functions := cls._fit_functions_cache.get(key)) is None:
                logger.debug('Compiling fit functions for %s', key)
                functions = cls.setup_fitter(params_shape, expvals_shape)
                cls._fit_functions_cache[key] = functions

        return functions

    @classmethod
    def setup_fitter(cls, params_shape: tuple[int, ...], expvals_shape: tuple[int, int]):
        def objective(params, xvals, meas_bases, initial_states, expvals, expvals_err, prep):
            xyzs = cls.unitary_params(params, xvals, npmod=jnp)
            unitaries = so3_cartesian(xyzs, npmod=jnp) @ prep
            r_elements = unitaries[:, meas_bases, initial_states]
            return jnp.sum(jnp.square((r_elements - expvals) / expvals_err))

        params = np.zeros(params_shape, dtype=float)
        args = (
            np.zeros(expvals_shape[0], dtype=float),  # xvals
            np.zeros(expvals_shape[1], dtype=int),  # meas_bases
            np.zeros(expvals_shape[1], dtype=int),  # initial_states
            np.zeros(expvals_shape, dtype=float),  # expvals
            np.zeros(expvals_shape, dtype=float),  # expvals_err
            np.zeros((3, 3), dtype=float)  # prep
        )
        in_axes = [0] + [None] * len(args)
        vobj = jax.jit(jax.vmap(objective, in_axes=in_axes)).lower(params, *args).compile()
        solver = jaxopt.GradientDescent(objective, maxiter=10000, tol=1.e-4)
        vsolve = jax.jit(jax.vmap(solver.run, in_axes=in_axes)).lower(params, *args).compile()
        hess = jax.jit(jax.hessian(objective)).lower(params[0], *args).compile()
        return vobj, vsolve, hess

    @classmethod
    def unitary_params(cls, fit_params: np.ndarray, xval: np.ndarray, npmod=np):
        raise NotImplementedError()


def make_ut_dataframe(experiment_data):
    """Helper function to construct a pandas DataFrame from unitary tomography results."""
    obs_keys = experiment_data.analysis_results('expvals_observed').value.keys()
    prep_params = experiment_data.analysis_results('prep_parameters').value
    unit_params = experiment_data.analysis_results('unitary_parameters').value
    data = []
    for state, prep in obs_keys:
        if prep == 'prep':
            source = prep_params
        else:
            source = unit_params
        datum = {'control': f'{state}{prep}'}
        for op, val in zip(['x', 'y', 'z'], source[state]):
            datum[op] = f'{val.n:.3f}±{val.std_dev:.3f}'
        data.append(datum)

    columns = ['control', 'x', 'y', 'z']
    return pd.DataFrame(data=data, columns=columns)
