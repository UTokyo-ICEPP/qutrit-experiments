"""Common operation functions for generating dummy data."""

from typing import Optional, Union
import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit_experiments.framework import BaseExperiment

from ..constants import DEFAULT_SHOTS

iq_centroids = np.array([[-1.e+7, -2.5e+7], [2.e+7, -3.e+7], [1.5e+7, -1.2e+7]])
iq_sigma = 4.e+6


def ef_memory(
    base_probs: np.ndarray,
    shots: int,
    num_qubits: Optional[int] = None,
    meas_return: str = 'single',
    states: tuple[int, int] = (0, 2)
) -> list[np.ndarray]:
    """Generate the memory ndarray from the probabilities for the base state.

    Args:
        base_probs: Probability of observing the first state of `states`.
            The flattened array determines the number of circuits.
        shots: Number of shots.
        num_qubits: Number of qubits.
        states: (0, 1), (1, 2), or (0, 2).

    Returns:
        A list (length num_circuits) of ndarrays with shape [shots, qubits, 2]
        (meas_level == 'single') or [qubits, 2] (meas_level == 'avg').
    """
    rng = np.random.default_rng()

    base_probs = base_probs.flatten()

    fluctuations = rng.standard_normal(base_probs.shape + (shots, 2)) * iq_sigma

    memory = []

    for prob, fluc in zip(base_probs, fluctuations):
        indices = rng.choice(states, size=shots, p=[prob, 1. - prob])
        iqvals = fluc + iq_centroids[indices]

        if num_qubits is None:
            iqvals = iqvals[:, None, :]
        else:
            iqvals = np.tile(iqvals[:, None, :], (1, num_qubits, 1))

        if meas_return == 'avg':
            iqvals = np.mean(iqvals, axis=0)

        memory.append(iqvals)

    return memory


def single_qubit_counts(
    probs: np.ndarray,
    shots: int,
    num_qubits: Optional[int] = None
) -> list[Counts]:
    rng = np.random.default_rng()
    c0s = rng.binomial(shots, probs)

    if num_qubits is None:
        counts = [Counts({'0': v, '1': shots - v}) for v in c0s.flatten()]
    else:
        counts = [Counts({'0' * num_qubits: v, '1' * num_qubits: shots - v})
                  for v in c0s.flatten()]

    return counts


def from_one_probs(
    experiment: BaseExperiment,
    one_probs: np.ndarray
) -> Union[list[np.ndarray], list[Counts]]:
    shots = experiment.run_options.get('shots', DEFAULT_SHOTS)
    num_qubits = 1
    if experiment.run_options.meas_level == MeasLevel.KERNELED:
        if experiment.experiment_options.discrimination_basis == '01':
            states = (0, 1)
        elif experiment.experiment_options.discrimination_basis == '02':
            states = (0, 2)
        else:
            states = (1, 2)
        meas_return = experiment.run_options.get('meas_return', 'avg')
        return ef_memory(one_probs, shots, num_qubits, meas_return, states=states)
    return single_qubit_counts(one_probs, shots, num_qubits)
