"""Common functions for running calibration programs."""
import logging
from typing import Any, Optional, Union
from qiskit_experiments.framework import BaseAnalysis, CompositeAnalysis, ExperimentData

from ..runners import ExperimentsRunner
from ..experiment_config import ExperimentConfigBase

LOG = logging.getLogger(__name__)


def run_experiment(
    runner: ExperimentsRunner,
    config: Union[str, ExperimentConfigBase],
    block_for_results: bool = True,
    analyze: bool = True,
    calibrate: bool = True,
    print_level: Optional[int] = None,
    force_resubmit: bool = False,
    plot_depth: int = 0,
    parallelize: bool = True,
    update_qubits: bool = False,
    save_data: bool = False
) -> ExperimentData:
    """Run an experiment."""
    runner_qubits = set(runner.qubits)

    exp_data = None
    if runner.saved_data_exists(config):
        exp_data = runner.load_data(config)
        if update_qubits:
            if (data_qubits := set(exp_data.metadata['physical_qubits'])) - runner_qubits:
                LOG.warning('Saved experiment data for %s has out-of-configuration qubits.',
                               config if isinstance(config, str) else config.exp_type)
            runner.qubits = data_qubits

    exp = runner.make_experiment(config)
    set_analysis_option(exp.analysis, 'plot', False, depth=plot_depth + 1)
    if not parallelize:
        set_analysis_option(exp.analysis, 'parallelize', 0)

    exp_data = runner.run_experiment(config, experiment=exp, block_for_results=block_for_results,
                                     analyze=analyze, calibrate=calibrate, print_level=print_level,
                                     exp_data=exp_data, force_resubmit=force_resubmit)

    if update_qubits:
        runner.qubits = runner_qubits

    if save_data:
        runner.program_data.setdefault('experiment_data', {})[exp_data.experiment_type] = exp_data

    return exp_data


def set_analysis_option(
    analysis: BaseAnalysis,
    option: str,
    value: Any,
    depth: int = 0,
    _current_depth: int = 0
) -> None:
    """Recursively set an option of an Analysis.

    In Qiskit Experiments >= 0.6 a similar operation is possible with
    set_options(..., broadcast=True) but it lacks the depth specification.

    Args:
        analysis: Analysis to set the option on.
        option: Option name.
        value: Option value.
        depth: Nesting depth of CompositeAnalyses to reach.
    """
    if _current_depth > depth:
        return

    if hasattr(analysis.options, option):
        analysis.set_options(**{option: value})

    if isinstance(analysis, CompositeAnalysis):
        for subanalysis in analysis.component_analysis():
            set_analysis_option(subanalysis, option, value, depth, _current_depth + 1)
