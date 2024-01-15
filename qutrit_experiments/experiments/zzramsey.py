"""Measurement of the qutrit-qubit ZZ Hamiltonian through SpectatorRamseyXY experiments with three
control states."""
from collections.abc import Sequence
from typing import Any, Optional
from matplotlib.figure import Figure
import numpy as np
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from ..util.matplotlib import make_list_plot
from .spectator_ramsey import SpectatorRamseyXY

twopi = 2. * np.pi


class QutritZZRamsey(BatchExperiment):
    """Measurement of the qutrit-qubit ZZ Hamiltonian through SpectatorRamseyXY experiments with
    three control states."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Optional[list] = None,
        osc_freq: Optional[float] = None,
        delay_schedule: Optional[ScheduleBlock] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []
        for control_state in range(3):
            exp = SpectatorRamseyXY(physical_qubits, control_state, delays=delays,
                                    osc_freq=osc_freq, delay_schedule=delay_schedule,
                                    backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=QutritZZRamseyAnalysis(analyses))

        self.extra_metadata = extra_metadata or {}

    def _metadata(self):
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)

        return metadata


class QutritZZRamseyAnalysis(CompoundAnalysis):
    """Analysis for QutritZZRamsey."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        for analysis in self._analyses:
            analysis.options.plot = self.options.plot

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata["component_child_index"]

        omega_zs_by_state = []

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            omega_zs_by_state.append(child_data.analysis_results('freq').value * twopi)

        omega_zs_by_state = np.array(omega_zs_by_state)

        op_to_state = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]])
        # Multiply by factor two to obtain omega_[Izζ]z
        omega_zs = (np.linalg.inv(op_to_state) @ omega_zs_by_state) * 2.

        analysis_results.append(
            AnalysisResultData(name='omega_zs', value=omega_zs)
        )

        if self.options.plot:
            figures.append(
                make_list_plot(experiment_data,
                               title_fn=lambda idx: fr'Control: $|{control_state}\rangle$')
            )

        return analysis_results, figures
