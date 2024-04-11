from typing import Optional
import numpy as np
from qiskit_experiments.data_processing import DataProcessor
from .multi_probability import MultiProbability, SerializeMultiProbability
from .readout_mitigation import ReadoutMitigation


def get_ternary_data_processor(
    assignment_matrix: Optional[np.ndarray] = None,
    include_invalid: bool = False,
    serialize: bool = False
) -> DataProcessor:
    nodes = []
    if assignment_matrix is not None:
        nodes.append(ReadoutMitigation(assignment_matrix))
    outcomes = ['10', '01', '11']
    if include_invalid:
        outcomes += ['00']
    nodes.append(MultiProbability(outcomes, [0.5] * len(outcomes)))
    if serialize:
        nodes.append(SerializeMultiProbability(outcomes))
    return DataProcessor('counts', nodes)
