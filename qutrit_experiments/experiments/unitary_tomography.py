"""Tomography of a unitary through measurements of the upper triangle of the rotation matrix."""
from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)

from ..experiment_mixins import MapToPhysicalQubits
from ..util.unitary_fit import fit_unitary

twopi = 2. * np.pi


class UnitaryTomography(MapToPhysicalQubits, BaseExperiment):
    """Tomography of a unitary through measurements of the upper triangle of the rotation matrix."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit = None
        options.pre_circuit = None
        options.measured_logical_qubit = 0
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
        axes = ['x', 'y', 'z']
        circuits = []
        for idx_i, initial_state in enumerate(axes):
            template = QuantumCircuit(num_qubits, 1)
            if (pre_circuit := self.experiment_options.pre_circuit) is not None:
                template.compose(pre_circuit, inplace=True)
                template.barrier()
            if initial_state != 'z':
                template.sx(meas_qubit)
            if initial_state == 'x':
                template.rz(np.pi / 2., meas_qubit)
            elif initial_state == 'y':
                template.rz(-np.pi, meas_qubit)
            template.barrier()
            template.compose(self.experiment_options.circuit, inplace=True)
            template.barrier()
            for idx_m in range(idx_i + 1):
                meas_basis = axes[idx_m]
                circuit = template.copy()
                if meas_basis == 'x':
                    circuit.rz(np.pi / 2., meas_qubit)
                if meas_basis != 'z':
                    circuit.sx(meas_qubit)
                circuit.measure(meas_qubit, 0)
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


class UnitaryTomographyAnalysis(BaseAnalysis):
    """Analysis for UnitaryTomography."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = None
        options.plot = True
        return options

    def _run_analysis(
        self,
        experiment_data: ExperimentData,
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        popt, state, observed, predicted, figure = fit_unitary(
            experiment_data.data(),
            data_processor=self.options.data_processor,
            plot=self.options.plot
        )

        analysis_results = [
            AnalysisResultData(name='unitary_fit_state', value=state),
            AnalysisResultData(name='unitary_fit_params', value=popt),
            AnalysisResultData(name='expvals_observed', value=observed),
            AnalysisResultData(name='expvals_predicted', value=predicted)
        ]
        if figure is not None:
            return analysis_results, [figure]
        return analysis_results, []
