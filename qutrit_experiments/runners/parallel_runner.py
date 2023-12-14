"""Driver for parallel single-qubit experiments."""

import copy
import logging
import math
import re
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Optional, Union
from matplotlib.figure import Figure
import numpy as np

from qiskit_experiments.calibration_management import BaseCalibrationExperiment
from qiskit_experiments.framework import BackendData, BaseExperiment, ExperimentData
from qiskit_experiments.framework.matplotlib import default_figure_canvas

from .experiments_runner import ExperimentsRunner, display, print_details, print_summary
from ..framework_overrides.batch_experiment import BatchExperiment
from ..framework_overrides.parallel_experiment import ParallelExperiment
from ..util.matplotlib import copy_axes

if TYPE_CHECKING:
    from qiskit.providers import Backend
    from qiskit_experiments.calibration_management import Calibrations
    from qiskit_ibm_runtime import Session

    from ..experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)


class ParallelRunner(ExperimentsRunner):
    """Driver for parallel single-qubit experiments."""
    def __init__(
        self,
        backend: 'Backend',
        active_qubits: Optional[Sequence[int]] = None,
        calibrations: Optional['Calibrations'] = None,
        data_dir: Optional[str] = None,
        read_only: bool = False,
        runtime_session: Optional['Session'] = None
    ):
        super().__init__(backend, calibrations=calibrations, data_dir=data_dir, read_only=read_only,
                         runtime_session=runtime_session)

        self.scheduling_triggers = ['set_f12']

        self.set_qubit_grouping(active_qubits=active_qubits)

        self.num_analysis_procs = -1

    @property
    def active_qubits(self):
        return set(sum(self.qubit_grouping, []))

    def set_qubit_grouping(self, active_qubits: Optional[Sequence[int]] = None):
        if active_qubits is None:
            active_qubits = list(range(self.backend.num_qubits))

        # Determine the grouping based on the topology + mask
        coupling_map = BackendData(self.backend).coupling_map
        # substitute for BackendV2 backend.coupling_map.neighbors
        def neighbors(qubit):
            return [cpl[1] for cpl in coupling_map if cpl[0] == qubit]

        self.qubit_grouping = []

        for qubit in active_qubits:
            group_candidates = []
            for group in self.qubit_grouping:
                if any((neighbor in group) for neighbor in neighbors(qubit)):
                    continue

                group_candidates.append(group)

            if len(group_candidates) == 0:
                self.qubit_grouping.append([qubit])
            else:
                igroup = np.argmin([len(group) for group in group_candidates])
                group_candidates[igroup].append(qubit)

    def make_experiment(
        self,
        config: 'ExperimentConfig'
    ) -> BatchExperiment:
        """Create a BatchExperiment of ParallelExperiment of exp_cls."""
        subexps = {}
        for qubit in self.active_qubits:
            qubit_config = copy.copy(config)
            qubit_config.physical_qubits = [qubit]
            qubit_config.exp_type = f'{config.exp_type}-q{qubit}'
            subexps[qubit] = super().make_experiment(qubit_config)

        return self.compose_experiment(subexps)

    def run_experiment(
        self,
        config: 'ExperimentConfig',
        experiment: Optional[BatchExperiment] = None,
        postprocess: bool = True,
        analyze: bool = True,
        calibrate: Union[bool, Callable[[ExperimentData], bool]] = True,
        print_level: int = 2
    ) -> ExperimentData:
        """Run the batch experiment."""
        if experiment is None:
            experiment = self.make_experiment(config)

        exp_data = super().run_experiment(config, experiment, postprocess=postprocess,
                                          analyze=analyze, calibrate=False, print_level=0)

        if not analyze or experiment.analysis is None:
            return exp_data

        if calibrate and issubclass(config.cls, BaseCalibrationExperiment):
            qubit_experiments = self.decompose_experiment(experiment)
            qubit_data = self.decompose_data(exp_data)

            if callable(calibrate):
                # Run selective calibration
                for qubit, child_data in list(qubit_data.items()):
                    try:
                        if calibrate(child_data):
                            continue
                    except Exception as ex:
                        logger.warning('Calibration criterion could not be evaluated for qubit %d:'
                                       ' %s', qubit, ex)
                    qubit_experiments.pop(qubit)
                    qubit_data.pop(qubit)

                self.set_qubit_grouping(active_qubits=list(sorted(qubit_data.keys())))

            self.update_calibrations(self.compose_data(qubit_data),
                                     self.compose_experiment(qubit_experiments))

        # Make and show the plots
        figures = self.consolidate_figures(qubit_data)
        exp_data.add_figures(figures)

        if print_level:
            for figure in figures:
                display(figure)
        if print_level >= 2:
            for qubit, child_data in qubit_data.items():
                logger.info('Qubit %d', qubit)
                if print_level == 2:
                    print_summary(child_data)
                else:
                    print_details(child_data)

        return exp_data

    def consolidate_figures(self, qubit_data: dict[int, ExperimentData]) -> list[Figure]:
        """Extract the figure objects from the experiment data and combine them in one figure.

        The consolidated figure is intended to be a quick-glance summary and therefore only takes
        the first axes of the subanalysis figures.
        """
        nrow = math.floor(math.sqrt(self._backend.num_qubits))
        ncol = math.ceil(math.sqrt(self._backend.num_qubits))
        if nrow * ncol > self._backend.num_qubits:
            nrow += 1

        figures = []

        for qubit, subdata in qubit_data.items():
            if not figures and subdata.figure_names:
                for _ in range(len(subdata.figure_names)):
                    figure = Figure()
                    default_figure_canvas(figure)
                    figure.set_figheight(2. * nrow)
                    figure.set_figwidth(2. * ncol)
                    figure.subplots(nrow, ncol)
                    figures.append(figure)

            for ifig in range(len(subdata.figure_names)):
                subax = figures[ifig].axes[qubit]
                origax = subdata.figure(ifig).figure.axes[0]

                copy_axes(origax, subax)

                # if qubit == self._backend.num_qubits // 2:
                #     label = origax.yaxis.get_label()
                #     subax.yaxis.set_label_text(label.get_text(), fontsize=label.get_fontsize())

                if qubit == self._backend.num_qubits - 1:
                    label = origax.xaxis.get_label()
                    subax.xaxis.set_label_text(label.get_text(), fontsize=label.get_fontsize())

        for figure in figures:
            figure.set_tight_layout(True)

        return figures

    def compose_experiment(
        self,
        experiments: dict[int, BaseExperiment]
    ) -> BatchExperiment:
        """Compose a batch experiment of parallel experiments from single-qubit experiments."""
        exp_type = re.sub(r'-q[0-9]+$', '', list(experiments.values())[0].experiment_type)

        parallel_exps = []
        for igroup, qubit_group in enumerate(self.qubit_grouping):
            # Make a ParallelExperiment
            subexps = [experiments[qubit] for qubit in qubit_group
                       if qubit in experiments]
            if not subexps:
                continue

            par_exp = ParallelExperiment(subexps, backend=self._backend)
            par_exp._type = f'{exp_type}-group{igroup}'
            par_exp.set_run_options(**subexps[0].run_options.__dict__)
            parallel_exps.append(par_exp)

        if not parallel_exps:
            return BatchExperiment([], backend=self.backend)

        experiment = BatchExperiment(parallel_exps, backend=self.backend)
        experiment._type = exp_type
        run_options_dict = parallel_exps[0].run_options.__dict__
        experiment.set_run_options(**run_options_dict)
        experiment.analysis.options.parallelize = self.num_analysis_procs
        experiment.analysis.options.ignore_failed = True

        return experiment

    def compose_data(
        self,
        child_data: dict[int, ExperimentData]
    ) -> ExperimentData:
        """Compose an ExperimentData from a list of child data."""
        service = list(child_data.values())[0].service

        parallel_data = []
        for qubit_group in self.qubit_grouping:
            # Make a ParallelExperiment
            subdata = [child_data[qubit] for qubit in qubit_group
                       if qubit in child_data]
            if not subdata:
                continue

            par_data = ExperimentData(backend=self.backend, service=service,
                                      child_data=subdata)
            parallel_data.append(par_data)

        return ExperimentData(backend=self.backend, service=service, child_data=parallel_data)

    def decompose_experiment(
        self,
        experiment: BatchExperiment,
        qubit_grouping: Optional[list[list[int]]] = None
    ) -> dict[int, BaseExperiment]:
        if qubit_grouping is None:
            active_qubits = self.active_qubits
        else:
            active_qubits = sum(qubit_grouping, [])

        flattened = {exp.physical_qubits[0]: exp
                     for par_exp in experiment.component_experiment()
                     for exp in par_exp.component_experiment()}
        return {qubit: flattened.get(qubit) for qubit in active_qubits}

    def decompose_data(
        self,
        experiment_data: ExperimentData,
        qubit_grouping: Optional[list[list[int]]] = None
    ) -> dict[int, ExperimentData]:
        if qubit_grouping is None:
            active_qubits = self.active_qubits
        else:
            active_qubits = sum(qubit_grouping, [])

        flattened = {data.metadata['physical_qubits'][0]: data
                     for par_data in experiment_data.child_data()
                     for data in par_data.child_data()}
        return {qubit: flattened.get(qubit) for qubit in active_qubits}
