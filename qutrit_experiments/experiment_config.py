"""Container for experiment definition and global experiment lists."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from qiskit_experiments.framework import BaseExperiment
    from ..common.postprocessed_experiment_data import PostProcessor


@dataclass
class ExperimentConfig:
    cls: type['BaseExperiment']
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
postprocessors = {}
experiment_products = {}

def register_exp(
    function: Optional[Callable[['ExperimentsRunner'], ExperimentConfig]] = None,
    *,
    product: Optional[str] = None
):
    if product is not None:
        def register_exp_and_product(function):
            experiment_products[function.__name__] = product
            return register_exp(function)

        return register_exp_and_product

    def registered_exp(runner):
        config = function(runner)
        config.exp_type = function.__name__
        return config

    experiments[function.__name__] = registered_exp
    return registered_exp


def register_post(function: Callable[['ExperimentsRunner'], None]):
    postprocessors[function.__name__] = function
    return function
