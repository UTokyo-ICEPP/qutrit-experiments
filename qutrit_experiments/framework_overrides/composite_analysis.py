"""Override of qiskit_experiments.framework.composite.composite_analysis."""

import logging
import sys
import time
import traceback
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import Pipe, Process, cpu_count
from multiprocessing.connection import Connection
from threading import Lock
from typing import Any, Optional, Union
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (AnalysisResult, AnalysisStatus, BaseAnalysis,
                                          CompositeAnalysis as CompositeAnalysisOrig,
                                          ExperimentData, FigureData, Options)

from ..framework.child_data import set_child_data_structure
from ..framework.threaded_analysis import NO_THREAD, ThreadedAnalysis

logger = logging.getLogger(__name__)


class CompositeAnalysis(CompositeAnalysisOrig):
    """CompositeAnalysis with modified functionalities.

    Modifications:
    - Parallelized the subanalyses with processes. Matplotlib is not thread-safe, so running the
      subanalyses with plots on threads can sometimes lead to weird errors.
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.parallelize = -1
        options.ignore_failed = False
        options.clear_results = True
        return options

    @staticmethod
    def _run_sub_analysis(
        analysis: BaseAnalysis,
        experiment_data: ExperimentData,
        thread_output: Optional[Any] = NO_THREAD,
        data_deserialized: bool = False
    ) -> tuple[
        AnalysisStatus,
        Union[
            tuple[list[AnalysisResult], list[FigureData]],
            tuple[Exception, str]
        ]
    ]:
        """Run a non-composite subanalysis on a component data."""
        # Bugfix
        if data_deserialized:
            experiment_data = experiment_data.copy(copy_results=False)
        else:
            experiment_data._clear_results()

        # copied from run_analysis in BaseAnalysis.run
        try:
            experiment_components = analysis._get_experiment_components(experiment_data)
            # making new analysis
            if thread_output is NO_THREAD:
                results, figures = analysis._run_analysis(experiment_data)
            else:
                results, figures = analysis._run_analysis_unthreaded(experiment_data, thread_output)
            # Add components
            analysis_results = [
                analysis._format_analysis_result(
                    result, experiment_data.experiment_id, experiment_components
                )
                for result in results
            ]
            # Update experiment data with analysis results
            if analysis_results:
                experiment_data.add_analysis_results(analysis_results)
            if figures:
                experiment_data.add_figures(figures, figure_names=analysis.options.figure_names)

            return AnalysisStatus.DONE, (experiment_data.analysis_results(),
                                         list(experiment_data._figures.values()))

        except Exception as ex:  # pylint: disable=broad-except
            return AnalysisStatus.ERROR, (ex, traceback.format_exc())

    @staticmethod
    def _run_sub_analysis_threaded(
        analysis: ThreadedAnalysis,
        experiment_data: ExperimentData,
    ) -> Any:
        """Run the threaded part of a ThreadedAnalysis."""
        return analysis._run_analysis_threaded(experiment_data)

    @staticmethod
    def _run_sub_composite_postanalysis(
        analysis: CompositeAnalysisOrig,
        parent_data: ExperimentData,
        parent_task_id: tuple[int, ...],
        component_data: list[ExperimentData],
        futures: dict[tuple[int, ...], Future],
        lock: Optional[Lock] = None
    ) -> tuple[list[AnalysisResult], list[FigureData]]:
        """Run the post-analysis for a composite subanalysis.

        This function is executed as a thread. If the parallelize option is on (determined by the
        existence of a lock), a process is dispatched to circumvent the GIL. If the component
        analyses of the subanalysis are themselves composite, the function is called recursively.
        """
        logger.debug('run_sub_composite parent task id %s num analysis %d num data %d',
                     parent_task_id, len(analysis.component_analysis()), len(component_data))
        for itask, (subanalysis, subdata) in enumerate(zip(analysis.component_analysis(),
                                                           component_data)):
            if not isinstance(subanalysis, CompositeAnalysisOrig):
                continue

            task_id = parent_task_id + (itask,)
            try:
                future = futures[task_id]
            except KeyError:
                # clear_results = False and this postanalysis has been done already
                continue

            logger.debug('Waiting for results from analysis of task %s', task_id)
            task_result = future.result()
            if isinstance(task_result, Exception):
                logger.debug('Received exception from %s:', task_id)
                if analysis.options.get('ignore_failed', False):
                    logger.warning('Ignoring postanalysis failure for %s:', task_id)
                    traceback.print_exception(task_result)
                    continue

                return task_result

            analysis_results, figures = task_result
            logger.debug('Received results from %s: %s', task_id, analysis_results)

            subdata._clear_results()
            if analysis_results:
                subdata.add_analysis_results(analysis_results)
            if figures:
                subdata.add_figures(figures, figure_names=subanalysis.options.figure_names)

        if lock is not None:
            # Parallelizing the postanalyses - run as a subprocess to circumvent the GIL
            logger.debug('Starting a subprocess for the analysis of experiment id %s',
                         parent_task_id)
            start = time.time()

            if hasattr(analysis, '_run_additional_analysis_threaded'):
                try:
                    thread_output = analysis._run_additional_analysis_threaded(parent_data)
                except Exception as exc:
                    if analysis.options.get('ignore_failed', False):
                        logger.warning('Ignoring postanalysis failure for %s:', parent_task_id)
                        traceback.print_exception(exc)
                        return [], []
                    return exc
            else:
                thread_output = NO_THREAD

            conn1, conn2 = Pipe()
            proc = Process(target=CompositeAnalysis._postanalysis,
                           args=(analysis, parent_data, component_data, thread_output, conn2))
            proc.start()
            # Can't recv() directly because recv deserializes the figures and can trigger
            # concurrency problems with mpl
            conn1.poll(None)
            with lock:
                msg = conn1.recv()
            conn1.close()
            proc.join()
            logger.debug('%s finished in %.2f seconds.',
                         parent_data.experiment_id, time.time() - start)
        else:
            try:
                msg = CompositeAnalysis._postanalysis(analysis, parent_data, component_data)
            except Exception as exc:
                msg = exc

        if isinstance(msg, Exception) and analysis.options.get('ignore_failed', False):
            logger.warning('Ignoring postanalysis failure for %s:', parent_task_id)
            traceback.print_exception(msg)
            return [], []

        # (analysis_results, figures) or exception
        return msg

    @staticmethod
    def _postanalysis(
        analysis: CompositeAnalysisOrig,
        parent_data: ExperimentData,
        component_data: list[ExperimentData],
        thread_output: Optional[Any] = NO_THREAD,
        conn: Optional[Connection] = None
    ) -> tuple[list[AnalysisResult], list[FigureData]]:
        analysis_results, figures = [], []

        try:
            results = []
            if analysis._flatten_results:
                results, figures = analysis._combine_results(component_data)

            if thread_output is not NO_THREAD:
                results, figures = analysis._run_additional_analysis_unthreaded(parent_data,
                                                                                results, figures,
                                                                                thread_output)
            elif hasattr(analysis, '_run_additional_analysis'):
                results, figures = analysis._run_additional_analysis(parent_data, results, figures)

            experiment_components = analysis._get_experiment_components(parent_data)

            analysis_results = [
                analysis._format_analysis_result(result, parent_data.experiment_id,
                                                 experiment_components)
                for result in results
            ]
        except Exception as ex:
            if conn:
                conn.send(ex)
            raise ex
        else:
            if conn:
                conn.send((analysis_results, figures))
        finally:
            if conn:
                conn.close()

        return analysis_results, figures

    def _gather_tasks(
        self,
        analysis: CompositeAnalysisOrig,
        experiment_data: ExperimentData,
        task_list: list[tuple[BaseAnalysis, ExperimentData, tuple[int, ...]]],
        subdata_map: dict[tuple[BaseAnalysis, ExperimentData], list[ExperimentData]],
        parent_task_id: tuple[int, ...] = ()
    ):
        """Recursively collect all atomic analyses and identify analysis dependencies."""
        logger.debug('Setting child data structure for task %s..', parent_task_id)
        set_child_data_structure(experiment_data)
        logger.debug('Extracting component data for task %s..', parent_task_id)
        component_expdata = analysis._component_experiment_data(experiment_data)
        logger.debug('Extracted component data for task %s..', parent_task_id)

        if hasattr(analysis, '_set_subanalysis_options'):
            analysis._set_subanalysis_options(experiment_data)

        for itask, (sub_analysis, sub_data) in enumerate(zip(analysis._analyses,
                                                             component_expdata)):
            task_id = parent_task_id + (itask,)
            if isinstance(sub_analysis, CompositeAnalysisOrig):
                self._gather_tasks(sub_analysis, sub_data, task_list, subdata_map, task_id)
            elif len(sub_data.analysis_results()) == 0 or self.options.clear_results:
                logger.debug('Booked %s for data from qubits %s',
                             sub_analysis.__class__.__name__, sub_data.metadata['physical_qubits'])
                task_list.append((sub_analysis, sub_data, task_id))

        # A call to _component_experiment_data recreates subdata and/or clears the saved results
        # -> keep the list in a container
        subdata_map[(analysis, experiment_data, parent_task_id)] = component_expdata

    def _run_analysis(self, experiment_data: ExperimentData):
        task_list = []
        subdata_map = {}
        self._gather_tasks(self, experiment_data, task_list, subdata_map)
        task_ids = [task_id for _, _, task_id in task_list]
        analyses = {task_id: analysis for analysis, _, task_id in task_list}
        sub_data = {task_id: sub_data for _, sub_data, task_id in task_list}

        # Run the component analysis on each component data
        if (max_procs := self.options.parallelize) != 0:
            if max_procs < 0:
                max_procs = cpu_count() // 2

            logger.debug('Running the subanalyses in parallel (max %d processes)..', max_procs)

            start = time.time()

            thread_outputs = [NO_THREAD] * len(task_ids)
            with ThreadPoolExecutor(max_workers=max_procs) as executor:
                futures = {}
                for itask, (task_id, analysis) in enumerate(analyses.items()):
                    if isinstance(analysis, ThreadedAnalysis):
                        futures[itask] = executor.submit(analysis._run_analysis_threaded,
                                                         sub_data[task_id])
                for itask, future in futures.items():
                    thread_outputs[itask] = future.result()

            # Backend cannot be pickled, so unset the attribute temporarily.
            # We use a list here but the backend cannot possibly be different among subdata..
            # Also BaseAnalysis.run is overwritten somewhere, making analysis instances unpickable.
            # We therefore delete the attribute temporarily and call _run_analysis directly in
            # _run_sub_analysis.
            backends = []
            runfuncs = []
            for tid in task_ids:
                backends.append(sub_data[tid].backend)
                sub_data[tid].backend = None
                runfuncs.append(analyses[tid].run)
                analyses[tid].run = None
            try:
                with ProcessPoolExecutor(max_workers=max_procs) as executor:
                    all_results = executor.map(CompositeAnalysis._run_sub_analysis,
                                               analyses.values(),
                                               sub_data.values(),
                                               thread_outputs,
                                               [True] * len(task_ids))
            finally:
                # Restore the original states of the data and analyses
                for tid, backend, runfunc in zip(task_ids, backends, runfuncs):
                    sub_data[tid].backend = backend
                    analyses[tid].run = runfunc

            logger.debug('Done in %.3f seconds.', time.time() - start)

        else:
            all_results = map(CompositeAnalysis._run_sub_analysis,
                              analyses.values(),
                              sub_data.values())

        for (status, retval), task_id in zip(all_results, task_ids):
            if status == AnalysisStatus.ERROR:
                exc, stacktrace = retval
                sys.stderr.write(stacktrace)
                sys.stderr.flush()
                if self.options.ignore_failed:
                    logger.warning('Ignoring analysis failure for analysis %s:', task_id)
                else:
                    raise AnalysisError(f'Analysis failed for analysis {task_id}') from exc
            elif max_procs != 0:
                # Multiprocess -> need to insert results to the experiment data in this process
                sub_data[task_id]._clear_results()
                analysis_results, figures = retval
                if analysis_results:
                    sub_data[task_id].add_analysis_results(analysis_results)
                if figures:
                    sub_data[task_id].add_figures([f.figure for f in figures],
                                                   figure_names=[f.name for f in figures])

        # Combine the child data if the analysis requires flattening
        # Entries in subdata_map is innermost-first
        # -> Data hierarchy is accounted for by just iterating
        logger.debug('Combining composite subanalysis results')
        start = time.time()

        # We use the futures dict for bookkeeping sub-sub-..-analyses and therefore execute the
        # postanalysis function through a thread pool even when options.parallelize == 0
        futures = {}
        if self.options.parallelize != 0:
            lock = Lock()
        else:
            lock = None

        with ThreadPoolExecutor() as executor:
            for (analysis, parent_data, task_id), component_expdata in subdata_map.items():
                if not self.options.clear_results and len(parent_data.analysis_results()) != 0:
                    continue

                future = executor.submit(CompositeAnalysis._run_sub_composite_postanalysis,
                                         analysis, parent_data, task_id, component_expdata, futures,
                                         lock)
                futures[task_id] = future

                if self.options.parallelize == 0 and task_id != ():
                    # Wait for completion
                    result = future.result()
                    if (isinstance(result, Exception)
                        and analysis.options.get('ignore_failed', False)):
                        raise AnalysisError(f'Postanalysis failed for {task_id}') from result

        try:
            future = futures[()]
        except KeyError:
            # Postanalysis did not run because results already existed and clear_results was False
            return (experiment_data.analysis_results(),
                    [experiment_data.figures(name) for name in experiment_data.figure_names])

        result = future.result()
        if isinstance(result, Exception):
            if self.options.ignore_failed:
                logger.warning('Ignoring postanalysis failure')
                traceback.print_exception(result)
                return [], []

            raise AnalysisError('Postanalysis failed') from result

        logger.debug('Done in %.3f seconds.', time.time() - start)
        # analysis_results, figures
        return result
