from collections.abc import Sequence
import logging
from threading import Lock
from typing import Any, Optional, Union
import jax
import jax.numpy as jnp
import jaxopt
from matplotlib.figure import Figure
import numpy as np
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.providers import Backend, Options
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.database_service.exceptions import ExperimentEntryNotFound
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.framework.matplotlib import get_non_gui_ax
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...framework_overrides.batch_experiment import BatchExperiment
from ...framework_overrides.composite_analysis import CompositeAnalysis
from ...framework.compound_analysis import CompoundAnalysis
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
        circuit: QuantumCircuit,
        tomography_type: str = 'unitary',
        measure_preparations: bool = True,
        control_states: Sequence[int] = (0, 1, 2),
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        analysis_cls: Optional[type[CompoundAnalysis]] = None
    ):
        experiments = []
        for iexp in range(5 if measure_preparations else 3):
            control_state = iexp if iexp < 3 else iexp - 2
            if control_state not in control_states:
                continue

            channel = QuantumCircuit(2)
            post_circuit = None
            if control_state >= 1:
                channel.x(0)
            if control_state == 2:
                channel.append(X12Gate(), [0])
                post_circuit = QuantumCircuit(2)
                post_circuit.barrier()
                post_circuit.append(X12Gate(), [0])

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
                        "experiment_type": self._type,
                        "composite_metadata": [circuit.metadata],
                        "composite_index": [index],
                    }
                    batch_circuits.append(circuit)

            return batch_circuits


class QutritQubitTomographyAnalysis(CompoundAnalysis):
    """Analysis for QutritQubitTomography."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names = ['expectation_values']
        options.data_processor = None # Needed to have DP propagated to tomography analysis
        options.plot = True
        options.parallelize = 0 # This analysis is somehow faster done in serial
        options.prep_unitaries = {}
        options.maxiter = None
        options.tol = None
        return options

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        for an in self._analyses:
            if (val := self.options.maxiter):
                an.set_options(maxiter=val)
            if (val := self.options.tol):
                an.set_options(tol=val)

    def _run_additional_analysis(
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
                unitary = (so3_cartesian(-prep_params, npmod=unp)
                           @ so3_cartesian(unitary_parameters[control_state], npmod=unp))
                unitary_parameters[control_state] = so3_cartesian_params(unitary, npmod=unp)

        analysis_results.extend([
            AnalysisResultData(name='unitary_parameters', value=unitary_parameters),
            AnalysisResultData(name='expvals_observed', value=observeds),
            AnalysisResultData(name='expvals_predicted', value=predicteds)
        ])

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
        circuit: QuantumCircuit,
        param_name: Union[str, Sequence[str]],
        values: Union[Sequence[float], Sequence[Sequence[float]]],
        angle_param_name: Optional[Union[str, dict[str, str]]] = None,
        tomography_type: str = 'unitary',
        measure_preparations: bool = True,
        control_states: Sequence[int] = (0, 1, 2),
        backend: Optional[Backend] = None,
        analysis_cls: Optional[type[CompositeAnalysis]] = None
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
        except:
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
                    "experiment_type": self._type,
                    "composite_metadata": [template_circuit.metadata],
                    "composite_index": [iexp],
                }
                circuits.append(circuit)

            if self.component_experiment(iexp).num_experiments > len(self.control_states):
                for template_circuit in template_circuits[num_tomography_circuits:]:
                    circuit = template_circuit.copy()
                    circuit.metadata = {
                        "experiment_type": self._type,
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


class QutritQubitTomographyScanAnalysis(CompoundAnalysis):
    """Analysis for QutritQubitTomographyScan."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names = ['theta_x', 'theta_y', 'theta_z', 'chisq']
        options.data_processor = None # Needed to have DP propagated to tomography analysis
        options.parallelize_on_thread = False
        options.simul_fit = False
        options.plot = True
        options.parallelize = 0 # This analysis is somehow faster done in serial
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

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata['component_child_index']
        parameters = experiment_data.metadata['scan_parameters']
        scan_values = [np.array(v) for v in experiment_data.metadata['scan_values']]

        control_states = None
        unitaries = []
        observeds = []
        predicteds = []
        prep_params = None
        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            if not control_states:
                control_states = child_data.metadata['control_states']
            try:
                prep_params = child_data.analysis_results('prep_parameters').value
                unitary_params = child_data.analysis_results('raw_parameters').value
            except ExperimentEntryNotFound:
                unitary_params = child_data.analysis_results('unitary_parameters').value
            if prep_params is not None:
                for state, params in prep_params.items():
                    unitary = (so3_cartesian(-params, npmod=unp)
                               @ so3_cartesian(unitary_params[state], npmod=unp))
                    unitary_params[state] = so3_cartesian_params(unitary, npmod=unp)
            unitaries.append(unitary_params)
            observeds.append(child_data.analysis_results('expvals_observed').value)
            predicteds.append(child_data.analysis_results('expvals_predicted').value)

        unitaries = np.array([[u[c] for c in control_states] for u in unitaries])
        observeds = np.array([[o[(c, '')] for c in control_states] for o in observeds])
        predicteds = np.array([[p[(c, '')] for c in control_states] for p in predicteds])

        chisq = np.sum(
            np.square((unp.nominal_values(observeds) - predicteds)
                       / unp.std_devs(observeds)),
            axis=2
        )
        chisq /= observeds.shape[2]

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
            popt_ufloats = self.simultaneous_fit(experiment_data, unitaries,
                                                 self.options.data_processor)
            analysis_results.append(
                AnalysisResultData('simul_fit_params', value=popt_ufloats)
            )

        if self.options.plot:
            if self.options.simul_fit:
                xvals = experiment_data.metadata['scan_values'][0]
                x_interp = np.linspace(xvals[0], xvals[-1], 100)
                upopts = next(res.value for res in analysis_results
                              if res.name == 'simul_fit_params')
                popts = {control_state: unp.nominal_values(params)
                         for control_state, params in upopts.items()}
                xyz_preds = [self.unitary_params(popts[control_state], x_interp)
                             for control_state in control_states]

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
                for ic, control_state in enumerate(control_states):
                    plotter.set_series_data(
                        f'c{control_state}',
                        x_formatted=scan_values[0],
                        y_formatted=unp.nominal_values(unitaries[:, ic, iop]),
                        y_formatted_err=unp.std_devs(unitaries[:, ic, iop])
                    )
                    if self.options.simul_fit:
                        plotter.set_series_data(
                            x_interp=x_interp,
                            y_interp=xyz_preds[ic][:, iop]
                        )

                figures.append(plotter.figure())

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Circuit',
                ylabel='chisq'
            )
            for ic, control_state in enumerate(control_states):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=np.arange(len(scan_values[0])),
                    y_formatted=chisq[:, ic],
                    y_formatted_err=np.zeros_like(scan_values[0])
                )
            figures.append(plotter.figure())

        return analysis_results, figures

    def simultaneous_fit(
        self,
        experiment_data: ExperimentData,
        unitary_params: np.ndarray,
        data_processor: Optional[DataProcessor] = None
    ):
        # The scanned parameter must be the first in scan_values list
        xvals = np.array(experiment_data.metadata['scan_values'][0])
        # Use normalized xvals throughout
        xvals_norm = xvals[-1] - xvals[0]
        norm_xvals = xvals / xvals_norm

        # Use the first child data for control state information
        first = experiment_data.child_data(0)
        control_states = first.metadata['control_states']
        index_state_map = {idx: first.child_data(idx).metadata['control_state']
                            for idx in first.metadata['component_child_index']
                            if not first.child_data(idx).metadata.get('state_preparation', False)}
        try:
            prep_params = first.analysis_results('prep_parameters').value
        except ExperimentEntryNotFound:
            prep_unitaries = {}
        else:
            prep_unitaries = {state: so3_cartesian(unp.nominal_values(params))
                                for state, params in prep_params.items()}

        if data_processor is None:
            data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])

        axes = ['x', 'y', 'z']

        expvals = {state: [] for state in control_states}
        ixvals = {state: [] for state in control_states}
        initial_states = {state: [] for state in control_states}
        meas_bases = {state: [] for state in control_states}
        for datum, expval in zip(experiment_data.data(), data_processor(experiment_data.data())):
            scan_metadata = datum['metadata']
            qqt_metadata = scan_metadata['composite_metadata'][0]
            ut_metadata = qqt_metadata['composite_metadata'][0]
            if (control_state := index_state_map.get(qqt_metadata['composite_index'][0])) is None:
                continue

            ix = scan_metadata['composite_index'][0]
            expvals[control_state].append(expval)
            ixvals[control_state].append(ix)
            initial_states[control_state].append(axes.index(ut_metadata['initial_state']))
            meas_bases[control_state].append(axes.index(ut_metadata['meas_basis']))

        # Fit in the unitary space
        popt_ufloats = {}
        for control_state in control_states:
            indices = (ixvals[control_state], meas_bases[control_state], initial_states[control_state])
            expvals_n = unp.nominal_values(expvals[control_state])
            expvals_e = unp.std_devs(expvals[control_state])
            fit_args = (norm_xvals, indices, expvals_n, expvals_e / np.mean(expvals_e),
                        prep_unitaries.get(control_state, np.eye(3)))

            p0s = self._get_p0s(unitary_params, control_state)
            vobj, vsolve, hess = self.fit_functions(p0s, *fit_args)

            fit_result = vsolve(p0s, *fit_args)
            fvals = vobj(fit_result.params, *fit_args)
            iopt = np.argmin(fvals)
            popt = np.array(fit_result.params[iopt])
            logger.debug('Control state %d optimal parameters %s', control_state, popt)

            prep_unitary = prep_unitaries.get(control_state, np.eye(3))
            hess_args = (norm_xvals, indices, expvals_n, expvals_e, prep_unitary)
            pcov = np.linalg.inv(hess(popt, *hess_args))
            upopt = np.array(correlated_values(popt, pcov))
            self._postprocess_params(upopt, xvals_norm)
            logger.debug('Control state %d adjusted parameters %s', control_state, upopt)
            popt_ufloats[control_state] = upopt
            
        return popt_ufloats
    
    def _get_p0s(self, unitary_params: np.ndarray, control_state: int):
        raise NotImplementedError()
    
    def _postprocess_params(self, upopt: np.ndarray, norm: float):
        return
    
    _lock = Lock()
    _fit_functions_cache = {}

    @classmethod
    def fit_functions(cls, params, widths, indices, expvals, expvals_err, prep_unitary):
        key = (widths.shape[0], expvals.shape[0])
        with cls._lock:
            return cls._fit_functions_cache.setdefault(
                key,
                cls.setup_fitter(params, widths, indices, expvals, expvals_err, prep_unitary)
            )
    
    @classmethod
    def setup_fitter(cls, params, xvals, indices, expvals, expvals_err, prep_unitary):
        def objective(params, xvals, indices, expvals, expvals_err, prep_unitary):
            xyz = cls.unitary_params(params, xvals, npmod=jnp)
            unitaries = prep_unitary @ so3_cartesian(xyz, npmod=jnp)
            r_elements = unitaries[indices]
            return jnp.sum(jnp.square((r_elements - expvals) / expvals_err))

        key = (xvals.shape[0], expvals.shape[0])
        args = (xvals, indices, expvals, expvals_err, prep_unitary)
        in_axes = [0] + [None] * len(args)
        vobj = jax.jit(jax.vmap(objective, in_axes=in_axes)).lower(params, *args).compile()
        solver = jaxopt.GradientDescent(objective, maxiter=10000, tol=1.e-4)
        vsolve = jax.jit(jax.vmap(solver.run, in_axes=in_axes)).lower(params, *args).compile()
        hess = jax.jit(jax.hessian(objective)).lower(params[0], *args).compile()
        cls._fit_functions_cache[key] = (vobj, vsolve, hess)

    @classmethod
    def unitary_params(cls, fit_params, xval, npmod=np):
        raise NotImplementedError()
