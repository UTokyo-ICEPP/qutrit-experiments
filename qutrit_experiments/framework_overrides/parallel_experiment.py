"""Override of qiskit_experiments.framework.composite.parallel_experiment."""

from collections import defaultdict
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.qobj.utils import MeasLevel, MeasReturnType
from qiskit.result import Counts
from qiskit_experiments.framework import (BaseExperiment,
                                          ParallelExperiment as ParallelExperimentOrig)

from .composite_analysis import CompositeAnalysis


class ParallelExperiment(ParallelExperimentOrig):
    """ParallelExperiment with modified functionalities.

    Modifications:
    - Use the overridden CompositeAnalysis by default.
    """
    def __init__(
        self,
        experiments: list[BaseExperiment],
        backend: Optional['Backend'] = None,
        flatten_results: bool = False,
        analysis: Optional[CompositeAnalysis] = None,
    ):
        if analysis is None:
            analysis = CompositeAnalysis(
                [exp.analysis for exp in experiments], flatten_results=flatten_results
            )

        super().__init__(experiments, backend=backend, flatten_results=flatten_results,
                         analysis=analysis)
