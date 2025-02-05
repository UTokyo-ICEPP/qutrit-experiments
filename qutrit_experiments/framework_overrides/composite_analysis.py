"""CompositeAnalysis with process-based parallelization and postanalysis."""
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
import logging
from multiprocessing import cpu_count
import pickle
from threading import Lock
from typing import Any, Optional, Union
import warnings
from dateutil import tz
from matplotlib.figure import Figure
from qiskit_experiments.framework import AnalysisResultData, BaseAnalysis, ExperimentData, Options
from qiskit_experiments.framework.analysis_result_data import as_table_element
from qiskit_experiments.framework.base_analysis import _requires_copy
from qiskit_experiments.framework.composite import CompositeAnalysis as CompositeAnalysisOrig
from qiskit_experiments.framework.containers import ArtifactData, FigureData

from ..framework.threaded_analysis import ThreadedAnalysis

LOG = logging.getLogger(__name__)
MPL_FIGURE_LOCK = Lock()


class CompositeAnalysis(CompositeAnalysisOrig):
    """CompositeAnalysis with process-based parallelization and postanalysis."""
    @staticmethod
    def add_results(
        analysis: BaseAnalysis,
        expdata: ExperimentData,
        results: list[Union[AnalysisResultData, ArtifactData]],
        figures: list[Figure]
    ):
        for result in results:
            if isinstance(result, AnalysisResultData):
                # Populate missing data fields
                if not result.experiment_id:
                    result.experiment_id = expdata.experiment_id
                if not result.experiment:
                    result.experiment = expdata.experiment_type
                if not result.device_components:
                    result.device_components = analysis._get_experiment_components(expdata)
                if not result.backend:
                    result.backend = expdata.backend_name
                if not result.created_time:
                    result.created_time = datetime.now(tz.tzlocal())
                if not result.run_time:
                    result.run_time = expdata.running_time

                # To canonical kwargs to add to the analysis table.
                table_format = as_table_element(result)

                # Remove result_id to make sure the id is unique in the scope of the container.
                # This will let the container generate a unique id.
                del table_format["result_id"]

                expdata.add_analysis_results(**table_format)
            elif isinstance(result, ArtifactData):
                if not result.experiment_id:
                    result.experiment_id = expdata.experiment_id
                if not result.device_components:
                    result.device_components = analysis._get_experiment_components(expdata)
                if not result.experiment:
                    result.experiment = expdata.experiment_type
                expdata.add_artifacts(result)
            else:
                raise TypeError(
                    f"Invalid object type {result.__class__.__name__} for analysis results. "
                    "This data cannot be stored in the experiment data."
                )

        figure_to_add = []
        for figure in figures:
            if not isinstance(figure, FigureData):
                qubits_repr = "_".join(
                    map(str, expdata.metadata.get("device_components", [])[:5])
                )
                short_id = expdata.experiment_id[:8]
                figure = FigureData(
                    figure=figure,
                    name=f"{expdata.experiment_type}_{qubits_repr}_{short_id}.svg",
                )
            figure_to_add.append(figure)
        expdata.add_figures(figure_to_add, figure_names=analysis.options.figure_names)

    @staticmethod
    def pickle_figures(analysis: BaseAnalysis, fn_name: str, args: tuple[Any, ...]):
        """Run the analysis and pickle the figures before returning."""
        results, figures = getattr(analysis, fn_name)(*args)
        pickled_figures = [pickle.dumps(figure) for figure in figures]
        return results, pickled_figures

    @staticmethod
    def run_in_subprocess(analysis: BaseAnalysis, experiment_data: ExperimentData):
        if isinstance(analysis, ThreadedAnalysis):
            thread_output = analysis._run_analysis_threaded(experiment_data)
            fn_name = '_run_analysis_processable'
            args = (analysis, experiment_data, thread_output)
        else:
            fn_name = '_run_analysis'
            args = (experiment_data,)

        # - Backend cannot be pickled, so we unset the attribute temporarily.
        # - BaseAnalysis.run is overwritten for analyses attached to BaseCalibrationExperiments,
        #   which makes analysis instances unpickable. We therefore delete the attribute
        #   temporarily.
        # - If experiment_data has an outstanding analysis future (which is the case because this
        #   function is run as a future), __getstate__ called during process task submission raises
        #   a warning, which we silence.
        runfunc = analysis.run
        backend = experiment_data.backend
        experiment = experiment_data.experiment
        analysis.run = None
        experiment_data.backend = None
        experiment_data._experiment = None
        exp_data_LOG = logging.getLogger('qiskit_experiments.framework.experiment_data')
        current_level = exp_data_LOG.level
        exp_data_LOG.setLevel(logging.ERROR)
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                results, pickled_figures = executor.submit(
                    CompositeAnalysis.pickle_figures,
                    analysis,
                    fn_name,
                    args
                ).result()
        finally:
            analysis.run = runfunc
            experiment_data.backend = backend
            experiment_data._experiment = experiment
            exp_data_LOG.setLevel(current_level)

        with MPL_FIGURE_LOCK:
            figures = [pickle.loads(figdata) for figdata in pickled_figures]

        return results, figures

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.parallelize = -1
        return options

    @classmethod
    def _broadcast_option_keys(cls) -> list[str]:
        return ['outcome', 'data_processor', 'plot']

    def set_options(self, **fields):
        broadcast_options = {key: value for key, value in fields.items()
                             if key in self._broadcast_option_keys()}
        super().set_options(broadcast=True, **broadcast_options)
        for key in broadcast_options:
            fields.pop(key)

        super().set_options(**fields)

    def run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        **options,
    ) -> ExperimentData:
        """Overridden run() function."""
        if options:
            warnings.warn('Call-time options for CompositeAnalysis.run() are ignored.')

        if (max_workers := self.options.parallelize) == 0:
            return self._run(experiment_data, replace_results=replace_results)

        if max_workers < 0:
            max_workers = cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return self._run(experiment_data, replace_results=replace_results, executor=executor)

    def _run(
        self,
        experiment_data: ExperimentData,
        replace_results: bool = False,
        executor: Optional[ThreadPoolExecutor] = None
    ) -> ExperimentData:
        """Body of the run function which is also called by the parent composite analysis if there
        is one.

        We forgo the original run -> _run_analysis structure entirely and execute component analyses
        and postanalysis directly in this function. For nested composite analyses, we recursively
        call _run() to reach individual atomic (non-composite) component analyses. The executor
        argument is a ThreadPoolExecutor that is used to throttle the number of concurrent analyses.
        Each sub-ExperimentData should be given an analysis callback that waits for the future
        created by the executor.
        """
        # Make a new copy of experiment data if not updating results
        if not replace_results and _requires_copy(experiment_data):
            experiment_data = experiment_data.copy()

        if not self._flatten_results:
            # Initialize child components if they are not initialized
            # This only needs to be done if results are not being flattened
            self._add_child_data(experiment_data)

        # Return list of experiment data containers for each component experiment
        # containing the marginalized data from the composite experiment
        component_expdata = self._component_experiment_data(experiment_data)

        def add_callback(analysis, exp_data, future):
            def add_results(expdata):
                try:
                    results, figures = future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    LOG.error('Analysis failed with exception: %s', exc)
                else:
                    CompositeAnalysis.add_results(analysis, expdata, results, figures)

            exp_data.add_analysis_callback(add_results)

        # Run the component analysis on each component data
        # Since copy for replace result is handled at the parent level we always run with replace
        # result on component analysis
        for analysis, sub_expdata in zip(self._analyses, component_expdata):
            if isinstance(analysis, CompositeAnalysis):
                analysis._run(sub_expdata, replace_results=True, executor=executor)
            elif isinstance(analysis, CompositeAnalysisOrig):
                analysis.run(sub_expdata, replace_results=True)
            elif executor:
                future = executor.submit(CompositeAnalysis.run_in_subprocess, analysis, sub_expdata)
                add_callback(analysis, sub_expdata, future)
            else:
                results, figures = analysis._run_analysis(sub_expdata)
                CompositeAnalysis.add_results(analysis, sub_expdata, results, figures)

        if executor:
            # Submit the postanalysis on executor and add a blocker as analysis callback
            future = executor.submit(self._run_postanalysis, experiment_data, component_expdata)
            add_callback(self, experiment_data, future)
        else:
            # We could add this as an analysis callback, but then we'll risk creating uncontrolled
            # number of analysis threads with nested CompositeAnalyses
            results, figures = self._run_postanalysis(experiment_data, component_expdata)
            CompositeAnalysis.add_results(self, experiment_data, results, figures)

        return experiment_data

    _run_analysis: Optional[
        Callable[
            [ExperimentData],
            tuple[list[Union[AnalysisResultData, ArtifactData], list[Figure]]]
        ]
    ] = None
    """Override this attribute as a method if running combined analysis on top of the results of
    individual component analyses."""

    def _run_postanalysis(
        self,
        experiment_data: ExperimentData,
        component_expdata: list[ExperimentData]
    ) -> tuple[list[Union[AnalysisResultData, ArtifactData], list[Figure]]]:
        """Parent data formatting and combined analysis.

        This function is run as a thread if parallelize != 0.
        """
        # Clearing previous analysis data
        experiment_data._clear_results()

        if not experiment_data.data():
            warnings.warn("ExperimentData object data is empty.\n")

        for sub_expdata in component_expdata:
            sub_expdata.block_for_results()

        if self._flatten_results:
            results, figures = self._combine_results(component_expdata)
            for res in results:
                # Override experiment  ID because entries are flattened
                res.experiment_id = experiment_data.experiment_id

            CompositeAnalysis.add_results(self, experiment_data, results, figures)

        if not callable(self._run_analysis):
            return [], []

        if self.options.parallelize != 0:
            return CompositeAnalysis.run_in_subprocess(self, experiment_data)
        else:
            # pylint: disable-next=not-callable
            return self._run_analysis(experiment_data)
