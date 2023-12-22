"""Container for experiment definition and global experiment lists."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from qiskit_experiments.framework import BaseExperiment, ExperimentData
    from .framework.postprocessed_experiment_data import PostProcessor


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    cls: Optional[type['BaseExperiment']] = None
    physical_qubits: Optional[Sequence[int]] = None
    args: dict[str, Any] = field(default_factory=dict)
    experiment_options: dict[str, Any] = field(default_factory=dict)
    run_options: dict[str, Any] = field(default_factory=dict)
    postprocessors: list['PostProcessor'] = field(default_factory=list)
    subexperiments: Optional[list['ExperimentConfig']] = None
    analysis: bool = True
    analysis_options: dict[str, Any] = field(default_factory=dict)
    plot_depth: int = 0
    exp_type: str = field(init=False)

    def __post_init__(self):
        self.exp_type = self.cls.__name__


experiments = {}
postexperiments = {}
experiment_products = {}

def register_exp(
    function: Optional[Callable[['ExperimentsRunner'], ExperimentConfig]] = None,
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
    function: Optional[Callable[['ExperimentsRunner', 'ExperimentData'], None]] = None,
    *,
    exp_type: Optional[str] = None
):
    """Register a postexperiment function to the global postexperiments map."""
    if exp_type is None:
        exp_type = function.__name__

    postexperiments[exp_type] = function
    return function
