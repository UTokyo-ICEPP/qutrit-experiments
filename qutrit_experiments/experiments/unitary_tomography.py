"""Tomography of a unitary through measurements of the upper triangle of the rotation matrix."""
from collections.abc import Sequence
from itertools import product
from typing import Any, Optional
import matplotlib
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.framework import (AnalysisResultData, BaseExperiment, ExperimentData,
                                          Options)

from ..experiment_mixins import MapToPhysicalQubits
from ..framework.threaded_analysis import ThreadedAnalysis
from ..util.unitary_fit import fit_unitary, plot_unitary_fit

twopi = 2. * np.pi


class UnitaryTomography(MapToPhysicalQubits, BaseExperiment):
    """Tomography of a unitary through measurements of the rotation matrix."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit = None
        options.pre_circuit = None
        options.post_circuit = None
        options.measured_logical_qubit = 0
        options.prep_meas_bases = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        super().__init__(physical_qubits, backend=backend)
        self.set_experiment_options(circuit=circuit)
        self.analysis = UnitaryTomographyAnalysis()
        self.extra_metadata = extra_metadata or {}

    def circuits(self) -> list[QuantumCircuit]:
        num_qubits = len(self.physical_qubits)
        meas_qubit = self.experiment_options.measured_logical_qubit
        if (setups := self.experiment_options.prep_meas_bases) is None:
            setups = product(['x', 'y', 'z'], ['x', 'y', 'z'])

        post_circuit = self.experiment_options.post_circuit
        if post_circuit is None:
            num_clbits = 1
        else:
            num_clbits = max(1, post_circuit.num_clbits)

        template = QuantumCircuit(num_qubits, num_clbits)
        if (pre_circuit := self.experiment_options.pre_circuit) is not None:
            template.compose(pre_circuit, inplace=True)

        circuits = []
        for initial_state, meas_basis in setups:
            circuit = template.copy()
            if initial_state != 'z':
                circuit.sx(meas_qubit)
            if initial_state == 'x':
                circuit.rz(np.pi / 2., meas_qubit)
            elif initial_state == 'y':
                circuit.rz(-np.pi, meas_qubit)
            circuit.barrier()
            circuit.compose(self.experiment_options.circuit, inplace=True)
            circuit.barrier()
            if meas_basis == 'x':
                circuit.rz(np.pi / 2., meas_qubit)
            if meas_basis != 'z':
                circuit.sx(meas_qubit)
            circuit.measure(meas_qubit, 0)
            if post_circuit is not None:
                circuit.compose(post_circuit, inplace=True)
            circuit.metadata = {
                'initial_state': initial_state,
                'meas_basis': meas_basis
            }
            circuits.append(circuit)

        return circuits

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata


class UnitaryTomographyAnalysis(ThreadedAnalysis):
    """Analysis for UnitaryTomography."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = None
        options.plot = True
        options.maxiter = None
        options.tol = None

        return options

    def _run_analysis_threaded(self, experiment_data: ExperimentData) -> dict[str, Any]:
        options = {}
        if self.options.maxiter:
            options['maxiter'] = self.options.maxiter
        if self.options.tol:
            options['tol'] = self.options.tol
        fit_result = fit_unitary(experiment_data.data(), data_processor=self.options.data_processor,
                                 **options)
        return {'unitary_fit_result': fit_result}

    def _run_analysis_processable(
        self,
        experiment_data: ExperimentData,
        thread_output: dict[str, Any]
    ) -> tuple[list[AnalysisResultData], list[matplotlib.figure.Figure]]:
        popt_ufloats, state, predicted, fit_input = thread_output['unitary_fit_result']
        analysis_results = [
            AnalysisResultData(name='unitary_fit_state', value=state),
            AnalysisResultData(name='unitary_fit_params', value=popt_ufloats),
            AnalysisResultData(name='expvals_observed', value=fit_input[0]),
            AnalysisResultData(name='expvals_predicted', value=predicted)
        ]

        if self.options.plot:
            return analysis_results, [plot_unitary_fit(popt_ufloats, *fit_input)]
        return analysis_results, []
