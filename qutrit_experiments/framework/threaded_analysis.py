"""Analysis with a part run in a thread of the main process when run parallel under
CompositeAnalysis."""
from abc import abstractmethod
from typing import Any, Union
from matplotlib.figure import Figure
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis
from qiskit_experiments.framework.containers import ArtifactData
from qiskit_experiments.framework.experiment_data import ExperimentData


class ThreadedAnalysis(BaseAnalysis):
    """Analysis with a part run in a thread of the main process when run parallel under
    CompositeAnalysis."""
    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[Union[AnalysisResultData, ArtifactData]], list[Figure]]:
        thread_output = self._run_analysis_threaded(experiment_data)
        return self._run_analysis_processable(experiment_data, thread_output)

    @abstractmethod
    def _run_analysis_threaded(self, experiment_data: ExperimentData) -> dict[str, Any]:
        """Run the threaded part of the analysis and return the outputs with names."""

    @abstractmethod
    def _run_analysis_processable(
        self,
        experiment_data: ExperimentData,
        thread_output: dict[str, Any]
    ) -> tuple[list[Union[AnalysisResultData, ArtifactData]], list[Figure]]:
        """Run the rest of the analysis using the output from the threaded part."""
