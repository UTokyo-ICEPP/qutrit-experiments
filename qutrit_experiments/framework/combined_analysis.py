"""CompositeAnalysis with additional analysis on top."""
from abc import abstractmethod
from collections.abc import Callable
import logging
from typing import Any, Optional
from matplotlib.figure import Figure
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, ExperimentData

from ..framework_overrides.composite_analysis import CompositeAnalysis

logger = logging.getLogger(__name__)


class CombinedAnalysis(CompositeAnalysis):
    """CompositeAnalysis with additional analysis on top."""
    @classmethod
    def _broadcast_option_keys(cls) -> list[str]:
        return ['outcome', 'data_processor', 'plot']

    def __init__(
        self,
        analyses: list[BaseAnalysis],
        generate_figures: Optional[str] = "always"
    ):
        """Construct a new CombinedAnalysis.

        flatten_results=False is discouraged from QE>=0.6 and will possibly be deprecated at some
        point, but many of our analyses do rely on having the child data hierarchy at the moment
        (or at least they appear to be; further investigation needed).
        """
        super().__init__(analyses, flatten_results=False, generate_figures=generate_figures)

    def set_options(self, **fields):
        broadcast_options = {key: value for key, value in fields.items()
                             if key in self._broadcast_option_keys()}
        super().set_options(broadcast=True, **broadcast_options)
        for key in broadcast_options:
            fields.pop(key)

        super().set_options(**fields)

    @abstractmethod
    def _run_combined_analysis(
        self,
        experiment_data: 'ExperimentData',
        analysis_results: list['AnalysisResultData'],
        figures: list['Figure']
    ) -> tuple[list['AnalysisResultData'], list['Figure']]:
        pass

    _run_combined_analysis_threaded: Optional[Callable[['ExperimentData'], Any]] = None
    """Override this attribute as a method if a part of the additional analysis must be run in a
    thread of the main process (e.g. when making changes to the analysis object)."""

    _run_combined_analysis_unthreaded: Optional[
        Callable[
            ['ExperimentData', list['AnalysisResultData'], list['Figure'], Any],
            tuple[list['AnalysisResultData'], list['Figure']]
        ]
    ] = None
    """Override this attribute as a method if the result of _run_combined_analysis_threaded must
    be input to the additional analysis (possibly run in a separate process)."""
