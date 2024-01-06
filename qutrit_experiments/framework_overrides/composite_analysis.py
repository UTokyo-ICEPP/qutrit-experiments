"""Override of qiskit_experiments.framework.composite.composite_analysis."""

import logging
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import Pipe, Pool, Process, cpu_count
from multiprocessing.connection import Connection
from threading import Lock
from typing import Optional
from qiskit_experiments.exceptions import AnalysisError
from qiskit_experiments.framework import (AnalysisResult, AnalysisStatus, BaseAnalysis,
                                          CompositeAnalysis as CompositeAnalysisOrig,
                                          ExperimentData, FigureData, Options)

from ..framework.child_data import set_child_data_structure

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
        data_deserialized: bool = False
    ) -> tuple[AnalysisStatus, list[AnalysisResult], list[FigureData]]:
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
            results, figures = analysis._run_analysis(experiment_data)
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
            return AnalysisStatus.ERROR, ex

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
            try:
                analysis_results, figures = future.result()
            except Exception as exc:
                traceback.print_exception(exc)
                raise AnalysisError(f'Postanalysis failed for {task_id}') from exc

            logger.debug('Received results for %s: %s', task_id, analysis_results)

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
            conn1, conn2 = Pipe()
            proc = Process(target=CompositeAnalysis._postanalysis,
                           args=(analysis, parent_data, component_data, conn2))
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

            if isinstance(msg, Exception):
                if analysis.options.get('ignore_failed', False):
                    logger.warning('Postanalysis failed for %s', parent_task_id)
                    return [], []

                traceback.print_exception(type(msg), msg, None)
                raise AnalysisError(f'Postanalysis failed for {parent_task_id}') from msg

            analysis_results, figures = msg
        else:
            try:
                analysis_results, figures = CompositeAnalysis._postanalysis(analysis, parent_data,
                                                                            component_data)
            except Exception as exc:
                traceback.print_exception(exc)
                raise AnalysisError(f'Postanalysis failed for {parent_task_id}') from exc

        return analysis_results, figures

    @staticmethod
    def _postanalysis(
        analysis: CompositeAnalysisOrig,
        parent_data: ExperimentData,
        component_data: list[ExperimentData],
        conn: Optional[Connection] = None
    ) -> tuple[list[AnalysisResult], list[FigureData]]:
        analysis_results, figures = [], []

        try:
            results = []
            if analysis._flatten_results:
                results, figures = analysis._combine_results(component_data)

            if hasattr(analysis, '_run_additional_analysis'):
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

        for itask, (sub_analysis, sub_data) in enumerate(zip(analysis._analyses,
                                                             component_expdata)):
            task_id = parent_task_id + (itask,)
            if isinstance(sub_analysis, CompositeAnalysisOrig):
                self._gather_tasks(sub_analysis, sub_data, task_list, subdata_map, task_id)
            elif len(sub_data.analysis_results()) == 0 or self.options.clear_results:
                task_list.append((sub_analysis, sub_data, task_id))

        if hasattr(analysis, '_set_subanalysis_options'):
            analysis._set_subanalysis_options(experiment_data)

        # A call to _component_experiment_data recreates subdata and/or clears the saved results
        # -> keep the list in a container
        subdata_map[(analysis, experiment_data, parent_task_id)] = component_expdata

    def _run_analysis(self, experiment_data: ExperimentData):
        task_list = []
        subdata_map = {}
        self._gather_tasks(self, experiment_data, task_list, subdata_map)

        # Run the component analysis on each component data
        if (max_procs := self.options.parallelize) != 0:
            if max_procs < 0:
                max_procs = cpu_count() // 2

            logger.debug('Running the subanalyses in parallel (max %d processes)..', max_procs)

            start = time.time()

            # Runtime backend cannot be pickled, so unset the attribute temporarily
            # We use a list here but the backend cannot possibly be different among subdata..
            # Also BaseAnalysis.run is overwritten somewhere, making analysis instances unpickable.
            # We therefore delete the attribute temporarily and call _run_analysis directly in
            # _run_sub_analysis.
            backends = []
            runfuncs = []
            for an, sub_data, task_id in task_list:
                backends.append(sub_data.backend)
                sub_data.backend = None
                runfuncs.append(an.run)
                delattr(an, 'run')
            try:
                with Pool(processes=max_procs) as pool:
                    all_results = pool.starmap(CompositeAnalysis._run_sub_analysis,
                                               [(an, sub_data, True)
                                                for an, sub_data, _ in task_list])
            finally:
                # Restore the original states of the data and analyses
                for (an, sub_data, _), backend, runfunc in zip(task_list, backends, runfuncs):
                    sub_data.backend = backend
                    an.run = runfunc

            logger.debug('Done in %.3f seconds.', time.time() - start)

            for (status, retval), (_, sub_expdata, task_id) in zip(all_results, task_list):
                if status == AnalysisStatus.ERROR:
                    if self.options.ignore_failed:
                        logger.warning('Analysis failed for analysis %s', task_id)
                        traceback.print_exception(retval)
                    else:
                        raise AnalysisError(f'Analysis failed for analysis {task_id}') from retval
                else:
                    sub_expdata._clear_results()
                    if retval[0]:
                        sub_expdata.add_analysis_results(retval[0])
                    if retval[1]:
                        sub_expdata.add_figures([f.figure for f in retval[1]],
                                                figure_names=[f.name for f in retval[1]])
        else:
            for analysis, data, task_id in task_list:
                status, retval = CompositeAnalysis._run_sub_analysis(analysis, data)
                if status == AnalysisStatus.ERROR:
                    if self.options.ignore_failed:
                        logger.warning('Analysis failed for %s', task_id)
                        traceback.print_exception(retval)
                    else:
                        raise AnalysisError(f'Analysis failed for {task_id}') from retval

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
                    try:
                        future.result()
                    except Exception as exc:
                        traceback.print_exception(exc)
                        raise AnalysisError(f'Postanalysis failed for {task_id}') from exc

        try:
            future = futures[()]
        except KeyError:
            # Postanalysis did not run because results already existed and clear_results was False
            return (experiment_data.analysis_results(),
                    [experiment_data.figures(name) for name in experiment_data.figure_names])
        try:
            analysis_results, figures = future.result()
        except Exception as exc:
            traceback.print_exception(exc)
            raise AnalysisError('Postanalysis failed') from exc

        logger.debug('Done in %.3f seconds.', time.time() - start)

        return analysis_results, figures
