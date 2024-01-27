"""CompositeAnalysis with additional analysis on top."""
import logging
from typing import TYPE_CHECKING

from ..framework_overrides.composite_analysis import CompositeAnalysis

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from qiskit_experiments.framework import AnalysisResultData, ExperimentData


class CompoundAnalysis(CompositeAnalysis):
    """CompositeAnalysis with additional analysis on top."""
    @classmethod
    def _propagated_option_keys(cls) -> list[str]:
        return ['outcome', 'data_processor', 'plot']

    def _set_subanalysis_options(self, experiment_data: 'ExperimentData'):
        logger.debug('Setting options for %d subanalyses of %s', len(self._analyses), self.__class__)
        for key in self._propagated_option_keys():
            if (value := self.options.get(key)) is not None:
                for analysis in self._analyses:
                    analysis.set_options(**{key: value})

    def _run_additional_analysis(
        self,
        experiment_data: 'ExperimentData',
        analysis_results: list['AnalysisResultData'],
        figures: list['Figure']
    ) -> tuple[list['AnalysisResultData'], list['Figure']]:
        return [], []
