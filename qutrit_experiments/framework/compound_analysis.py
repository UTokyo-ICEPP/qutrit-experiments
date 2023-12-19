"""CompositeAnalysis with additional analysis on top."""

from typing import TYPE_CHECKING

from ..framework_overrides.composite_analysis import CompositeAnalysis

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qiskit_experiments.framework import AnalysisResultData, ExperimentData


class CompoundAnalysis(CompositeAnalysis):
    """CompositeAnalysis with additional analysis on top."""
    def _set_subanalysis_options(self, experiment_data: 'ExperimentData'):
        pass

    def _run_additional_analysis(
        self,
        experiment_data: 'ExperimentData',
        analysis_results: list['AnalysisResultData'],
        figures: list['Figure']
    ) -> tuple[list['AnalysisResultData'], list['Figure']]:
        return [], []
