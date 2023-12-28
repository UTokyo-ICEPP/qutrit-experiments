"""Container for experiment definition and global experiment lists."""

from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Optional
from qiskit_experiments.framework import BaseExperiment, ExperimentData

from .framework_overrides.batch_experiment import BatchExperiment
from .framework_overrides.parallel_experiment import ParallelExperiment


@dataclass
class ExperimentConfigBase:
    """Base class for experiment configuration."""
    _: KW_ONLY
    run_options: dict[str, Any] = field(default_factory=dict)
    analysis: bool = True
    analysis_options: dict[str, Any] = field(default_factory=dict)
    plot_depth: int = 0
    exp_type: str = ''


@dataclass
class ExperimentConfig(ExperimentConfigBase):
    """Experiment configuration."""
    cls: type[BaseExperiment]
    physical_qubits: Optional[Sequence[int]] = None
    _: KW_ONLY
    args: dict[str, Any] = field(default_factory=dict)
    experiment_options: dict[str, Any] = field(default_factory=dict)
    restless: bool = False


@dataclass
class CompositeExperimentConfig(ExperimentConfigBase):
    """Base class for composite experiment configuration."""
    subexperiments: list[ExperimentConfigBase]
    _cls: type[BaseExperiment] = field(init=False)

    @property
    def cls(self) -> type[BaseExperiment]:
        return self._cls


class BatchExperimentConfig(CompositeExperimentConfig):
    """Configuration of a BatchExperiment."""
    def __post_init__(self):
        self._cls = BatchExperiment


class ParallelExperimentConfig(CompositeExperimentConfig):
    """Configuration of a ParallelExperiment."""
    def __post_init__(self):
        self._cls = ParallelExperiment


experiments = {}
postexperiments = {}
experiment_products = {}

def register_exp(
    function: Optional[Callable[['ExperimentsRunner'], ExperimentConfigBase]] = None,
    *,
    exp_type: Optional[str] = None,
    product: Optional[str] = None
):
    """Register a configuration generator to the global experiments map."""
    if exp_type is None:
        exp_type = function.__name__

    if product is not None:
        def register_exp_and_product(function):
            experiment_products[exp_type] = product
            return register_exp(function)

        return register_exp_and_product

    def registered_exp(runner):
        config = function(runner)
        config.exp_type = exp_type
        return config

    experiments[exp_type] = registered_exp
    return registered_exp


def register_post(
    function: Optional[Callable[['ExperimentsRunner', ExperimentData], None]] = None,
    *,
    exp_type: Optional[str] = None
):
    """Register a postexperiment function to the global postexperiments map."""
    if exp_type is None:
        exp_type = function.__name__

    postexperiments[exp_type] = function
    return function
