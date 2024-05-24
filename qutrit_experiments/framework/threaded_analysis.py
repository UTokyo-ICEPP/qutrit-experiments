"""Analysis with a part run in a thread of the main process when run parallel under
CompositeAnalysis."""
from abc import abstractmethod
from enum import Enum, auto
from typing import Any
from matplotlib.figure import Figure
from qiskit_experiments.framework import BaseAnalysis
from qiskit_experiments.framework.analysis_result_data import AnalysisResultData
from qiskit_experiments.framework.experiment_data import ExperimentData


class ThreadedAnalysis(BaseAnalysis):
    """Analysis with a part run in a thread of the main process when run parallel under
    CompositeAnalysis."""

    class NoThreadingFlag(Enum):
        """Special constant to flag no theading."""
        NO_THREAD = auto()

    def _run_analysis(
        self,
        experiment_data: ExperimentData
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        thread_output = self._run_analysis_threaded(experiment_data)
        return self._run_analysis_unthreaded(experiment_data, thread_output)

    @abstractmethod
    def _run_analysis_threaded(self, experiment_data: ExperimentData) -> Any:
        pass

    @abstractmethod
    def _run_analysis_unthreaded(
        self,
        experiment_data: ExperimentData,
        thread_output: Any
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        pass


NO_THREAD = ThreadedAnalysis.NoThreadingFlag.NO_THREAD
