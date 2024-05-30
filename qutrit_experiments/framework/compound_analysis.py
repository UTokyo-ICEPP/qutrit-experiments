"""CompositeAnalysis with additional analysis on top."""
from abc import abstractmethod
from collections.abc import Callable
import copy
import logging
from typing import Any, Optional
from matplotlib.figure import Figure
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, ExperimentData

from ..framework_overrides.composite_analysis import CompositeAnalysis

logger = logging.getLogger(__name__)


class CompoundAnalysis(CompositeAnalysis):
    """CompositeAnalysis with additional analysis on top."""
    @classmethod
    def _propagated_option_keys(cls) -> list[str]:
        return ['outcome', 'data_processor', 'plot']

    def __init__(
        self,
        analyses: list[BaseAnalysis],
        generate_figures: Optional[str] = "always"
    ):
        """Construct a new CompoundAnalysis.

        flatten_results=False is discouraged from QE>=0.6 and will possibly be deprecated at some
        point, but many of our analyses do rely on having the child data hierarchy at the moment
        (or at least they appear to be; further investigation needed).
        """
        super().__init__(analyses, flatten_results=False, generate_figures=generate_figures)

    # pylint: disable-next=unused-argument
    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        logger.debug('Setting options for %d subanalyses of %s',
                     len(self._analyses), self.__class__)
        for key in self._propagated_option_keys():
            if (value := self.options.get(key)) is not None:
                for analysis in self._analyses:
                    analysis.set_options(**{key: copy.deepcopy(value)})

    @abstractmethod
    def _run_additional_analysis(
        self,
        experiment_data: 'ExperimentData',
        analysis_results: list['AnalysisResultData'],
        figures: list['Figure']
    ) -> tuple[list['AnalysisResultData'], list['Figure']]:
        pass

    _run_additional_analysis_threaded: Optional[Callable[['ExperimentData'], Any]] = None
    """Override this attribute as a method if a part of the additional analysis must be run in a
    thread of the main process (e.g. when making changes to the analysis object)."""

    _run_additional_analysis_unthreaded: Optional[
        Callable[
            ['ExperimentData', list['AnalysisResultData'], list['Figure'], Any],
            tuple[list['AnalysisResultData'], list['Figure']]
        ]
    ] = None
    """Override this attribute as a method if the result of _run_additional_analysis_threaded must
    be input to the additional analysis (possibly run in a separate process)."""
