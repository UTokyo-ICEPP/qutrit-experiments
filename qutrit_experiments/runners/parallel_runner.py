"""Driver for parallel single-qubit experiments."""

import copy
import logging
import math
from collections.abc import Callable, Sequence
from typing import Optional, Union
from matplotlib.figure import Figure
import numpy as np
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.framework import BackendData, BaseExperiment, ExperimentData
from qiskit_experiments.framework.matplotlib import default_figure_canvas
from qiskit_ibm_runtime import Session

from .experiments_runner import ExperimentsRunner, display, print_details, print_summary
from ..constants import RESTLESS_RUN_OPTIONS
from ..experiment_config import (BatchExperimentConfig, ExperimentConfig, ExperimentConfigBase,
                                 ParallelExperimentConfig, experiments)
from ..framework_overrides.batch_experiment import BatchExperiment
from ..util.matplotlib import copy_axes

logger = logging.getLogger(__name__)


class ParallelRunner(ExperimentsRunner):
    """Driver for parallel single-qubit experiments."""
    def __init__(
        self,
        backend: Backend,
        active_qubits: Optional[Sequence[int]] = None,
        calibrations: Optional[Calibrations] = None,
        data_dir: Optional[str] = None,
        read_only: bool = False,
        runtime_session: Optional[Session] = None
    ):
        super().__init__(backend, calibrations=calibrations, data_dir=data_dir, read_only=read_only,
                         runtime_session=runtime_session)

        self.set_qubit_grouping(active_qubits=active_qubits)
        self.num_analysis_procs = -1
        self.plot_all_qubits = False

    @property
    def active_qubits(self):
        return set(sum(self.qubit_grouping, []))

    def set_qubit_grouping(
        self,
        active_qubits: Optional[Sequence[int]] = None,
        max_group_size: int = 0
    ):
        if active_qubits is None:
            active_qubits = list(range(self.backend.num_qubits))

        # Determine the grouping based on the topology + mask
        coupling_map = BackendData(self.backend).coupling_map
        # substitute for BackendV2 backend.coupling_map.neighbors
        def neighbors(qubit):
            return (set(cpl[1] for cpl in coupling_map if cpl[0] == qubit)
                    | set(cpl[0] for cpl in coupling_map if cpl[1] == qubit))

        self.qubit_grouping = []

        for qubit in active_qubits:
            group_candidates = []
            for group in self.qubit_grouping:
                if (not any((neighbor in group) for neighbor in neighbors(qubit))
                    and (max_group_size <= 0 or len(group) < max_group_size)):
                    group_candidates.append(group)

            if len(group_candidates) == 0:
                self.qubit_grouping.append([qubit])
            else:
                igroup = np.argmin([len(group) for group in group_candidates])
                group_candidates[igroup].append(qubit)

    def make_batch_config(
        self,
        config: Union[ExperimentConfig, Callable[[ExperimentsRunner, int], ExperimentConfig]],
        exp_type: Optional[str] = None
    ) -> BatchExperimentConfig:
        """Create a configuration of a batch experiment of parallel experiments of experiments."""
        if not callable(config):
            exp_type = config.exp_type

        batch_conf = BatchExperimentConfig(exp_type=exp_type)
        composite_config = {}
        for igroup, qubit_group in enumerate(self.qubit_grouping):
            parallel_conf = ParallelExperimentConfig(exp_type=f'{exp_type}-group{igroup}')
            for qubit in qubit_group:
                if callable(config):
                    qubit_config = config(self, qubit)
                else:
                    qubit_config = copy.copy(config)
                    qubit_config.physical_qubits = [qubit]
                qubit_config.exp_type = f'{exp_type}-q{qubit}'

                if not composite_config:
                    composite_config['analysis'] = qubit_config.analysis
                    composite_config['run_options'] = dict(
                        qubit_config.cls._default_run_options().items()
                    )
                    composite_config['run_options'].update(qubit_config.run_options)
                    if qubit_config.restless:
                        composite_config['run_options'].update(RESTLESS_RUN_OPTIONS)
                    composite_config['max_circuits'] = qubit_config.experiment_options.get('max_circuits')
                    composite_config['calibration_criterion'] = qubit_config.calibration_criterion

                parallel_conf.subexperiments.append(qubit_config)

            # Run options on intermediate ParallelExperiments have no effect but warnings may be
            # issued if meas_level and meas_return_type are different between the parallelized
            # experiment and any container experiments, so we set the run options here too
            parallel_conf.analysis = composite_config['analysis']
            parallel_conf.run_options = composite_config['run_options']
            batch_conf.subexperiments.append(parallel_conf)

        batch_conf.analysis = composite_config['analysis']
        batch_conf.run_options = composite_config['run_options']
        batch_conf.experiment_options['max_circuits'] = composite_config['max_circuits']
        batch_conf.calibration_criterion = composite_config['calibration_criterion']
        return batch_conf

    def make_experiment(
        self,
        config: Union[str, ExperimentConfigBase]
    ) -> BatchExperiment:
        """Create a BatchExperiment of ParallelExperiment of exp_cls."""
        if isinstance(config, str):
            config = experiments[config](self)
        if isinstance(config, ExperimentConfig):
            batch_config = self.make_batch_config(config)
        else:
            batch_config = config

        experiment = super().make_experiment(batch_config)

        if config.analysis:
            experiment.analysis.set_options(
                parallelize=self.num_analysis_procs,
                ignore_failed=True
            )
            for parallel_exp in experiment.component_experiment():
                parallel_exp.analysis.set_options(ignore_failed=True)

        return experiment

    def run_experiment(
        self,
        config: Union[str, ExperimentConfig],
        experiment: Optional[BatchExperiment] = None,
        block_for_results: bool = True,
        analyze: bool = True,
        calibrate: bool = True,
        print_level: int = 2,
        force_resubmit: bool = False
    ) -> ExperimentData:
        """Run the batch experiment."""
        if isinstance(config, str):
            config = experiments[config](self)
        if isinstance(config, ExperimentConfig):
            batch_config = self.make_batch_config(config)
        else:
            batch_config = config

        if experiment is None:
            experiment = self.make_experiment(batch_config)

        exp_data = super().run_experiment(batch_config, experiment,
                                          block_for_results=block_for_results,
                                          analyze=analyze, calibrate=calibrate, print_level=0,
                                          force_resubmit=force_resubmit)

        if not analyze or experiment.analysis is None:
            return exp_data

        # Make and show the plots
        with exp_data._analysis_callbacks.lock:
            exp_data.add_analysis_callback(self.consolidate_figures)

        if block_for_results:
            exp_data.block_for_results()
            if print_level:
                for figure_name in exp_data.figure_names:
                    display(exp_data.figure(figure_name))
            if print_level >= 2:
                for qubit, child_data in self.decompose_data(exp_data).items():
                    logger.info('Qubit %d', qubit)
                    if print_level == 2:
                        print_summary(child_data)
                    else:
                        print_details(child_data)

        return exp_data

    def save_calibrations(
        self,
        exp_type: str,
        update_list: list[tuple[str, str, tuple[int, ...], bool]]
    ):
        """Save the calibrations and remove qubits that failed the calibration."""
        super().save_calibrations(exp_type, update_list)
        failed_qubits = set(qubits[0] for _, _, qubits, updated in update_list if not updated)
        active_qubits = self.active_qubits - failed_qubits
        self.set_qubit_grouping(active_qubits=active_qubits)

    def consolidate_figures(self, experiment_data: ExperimentData):
        """Extract the figure objects from the experiment data and combine them in one figure.

        The consolidated figure is intended to be a quick-glance summary and therefore only takes
        the first axes of the subanalysis figures.
        """
        if self.plot_all_qubits:
            num_qubits = self._backend.num_qubits
        else:
            num_qubits = len(self.active_qubits)
        nrow = math.floor(math.sqrt(num_qubits))
        ncol = math.ceil(math.sqrt(num_qubits))
        if nrow * ncol < num_qubits:
            nrow += 1

        figures = []

        for qubit, subdata in self.decompose_data(experiment_data).items():
            if subdata is None:
                # Discrepancy in the experiment data and active_qubits; implies an error somewhere
                # but we won't worry here
                continue

            if self.plot_all_qubits:
                iax = qubit
            else:
                iax = sorted(self.active_qubits).index(qubit)

            if not figures and subdata.figure_names:
                for _ in range(len(subdata.figure_names)):
                    figure = Figure()
                    default_figure_canvas(figure)
                    figure.set_figheight(2. * nrow)
                    figure.set_figwidth(2. * ncol)
                    figure.subplots(nrow, ncol)
                    figures.append(figure)

            for ifig in range(len(subdata.figure_names)):
                subax = figures[ifig].axes[iax]
                origax = subdata.figure(ifig).figure.axes[0]

                copy_axes(origax, subax)
                subax.set_title(f'Q{qubit}')

                # if qubit == self._backend.num_qubits // 2:
                #     label = origax.yaxis.get_label()
                #     subax.yaxis.set_label_text(label.get_text(), fontsize=label.get_fontsize())

                if qubit == self._backend.num_qubits - 1:
                    label = origax.xaxis.get_label()
                    subax.xaxis.set_label_text(label.get_text(), fontsize=label.get_fontsize())

        for figure in figures:
            figure.set_tight_layout(True)

        experiment_data.add_figures(figures)

    def decompose_experiment(
        self,
        experiment: BatchExperiment,
        active_qubits: Optional[Sequence[int]] = None
    ) -> dict[int, BaseExperiment]:
        if active_qubits is None:
            active_qubits = self.active_qubits

        flattened = {exp.physical_qubits[0]: exp
                     for par_exp in experiment.component_experiment()
                     for exp in par_exp.component_experiment()}
        return {qubit: flattened.get(qubit) for qubit in active_qubits}

    def decompose_data(
        self,
        experiment_data: ExperimentData,
        active_qubits: Optional[Sequence[int]] = None
    ) -> dict[int, ExperimentData]:
        if active_qubits is None:
            active_qubits = self.active_qubits

        flattened = {data.metadata['physical_qubits'][0]: data
                     for par_data in experiment_data.child_data()
                     for data in par_data.child_data()}
        return {qubit: flattened.get(qubit) for qubit in active_qubits}
