"""Probability action allowing multiple outcomes."""
from collections.abc import Sequence
from numbers import Number
from typing import Optional, Union
import numpy as np
from uncertainties import ufloat
from qiskit_experiments.data_processing.nodes import DataAction, TrainableDataAction
from qiskit_experiments.data_processing.exceptions import DataProcessorError
from qiskit_experiments.framework import Options


class MultiProbability(TrainableDataAction):
    """Probability of multiple outcomes."""
    @classmethod
    def _default_parameters(cls) -> Options:
        return Options(outcomes=None, alpha_prior=None)

    def __init__(
        self,
        outcomes: Optional[list[str]] = None,
        alpha_prior: Union[float, Sequence[float]] = 0.5,
        validate: bool = True,
    ):
        """Initialize a counts to probability data conversion.

        Ref. https://tminka.github.io/papers/minka-multinomial.pdf

        Args:
            outcomes: The bitstrings for which to return the probability and variance.
            alpha_prior: A prior Beta distribution parameter ``[`alpha0, alpha1]``.
                         If specified as float this will use the same value for
                         ``alpha0`` and``alpha1`` (Default: 0.5).
            validate: If set to False the DataAction will not validate its input.
        Raises:
            DataProcessorError: When the dimension of the prior and expected parameter vector
                do not match.
        """
        super().__init__(validate=validate)
        self._parameters.alpha_prior = alpha_prior
        if outcomes:
            self._parameters.outcomes = list(outcomes)

    def train(self, data: Union[dict, list[dict]]):
        params = self._parameters
        if not params.outcomes:
            bitstr_len = len(list(data[0].keys())[0])
            fmt = f'{{:0{bitstr_len}b}}'
            params.outcomes = [fmt.format(i) for i in range(2 ** bitstr_len)]

        if isinstance(params.alpha_prior, Number):
            params.alpha_prior = [params.alpha_prior] * len(params.outcomes)
        elif self._validate and len(params.alpha_prior) != len(params.outcomes):
            raise DataProcessorError(
                "Prior for probability node must be a float or pair of floats."
            )

    def _process(self, data: np.ndarray) -> np.ndarray:
        """Compute mean and standard error from the beta distribution.
        Args:
            data: A data array to process. This is a single numpy array containing
                all circuit results input to the data processor.
                This is usually an object data type containing Python dictionaries of
                count data keyed on the measured bitstring.
        Returns:
            The data that has been processed.
        """
        params = self._parameters
        probabilities = np.empty(data.size, dtype=object)

        for idx, counts_dict in enumerate(data):
            freqs = [counts_dict.get(outcome, 0) for outcome in params.outcomes]
            alpha_posterior = np.array([freq + alpha
                                        for freq, alpha in zip(freqs, params.alpha_prior)])
            alpha_sum = sum(alpha_posterior)

            p_mean = alpha_posterior / alpha_sum
            p_var = p_mean * (1 - p_mean) / (alpha_sum + 1)

            probabilities[idx] = {outcome: ufloat(nominal_value=m, std_dev=np.sqrt(v))
                                  for outcome, m, v in zip(params.outcomes, p_mean, p_var)}

        return probabilities


class SerializeMultiProbability(DataAction):
    """Serialize the result of MultiProbability."""
    def __init__(self, outcomes: list[str], validate: bool = True):
        super().__init__(validate=validate)
        self._outcomes = list(outcomes)

    def _process(self, data: np.ndarray) -> np.ndarray:
        nout = len(self._outcomes)
        probabilities = np.empty(data.size * nout, dtype=object)
        for idx, probs_dict in enumerate(data):
            for iout, outcome in enumerate(self._outcomes):
                probabilities[idx * nout + iout] = probs_dict[outcome]
        return probabilities
