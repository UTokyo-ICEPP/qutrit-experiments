"""Least-squares readout mitigator action."""
import numpy as np
import scipy.optimize as sciopt
from qiskit_experiments.data_processing.nodes import CountsAction


def _objective(x: np.ndarray, counts: np.ndarray, cal_matrix: np.ndarray) -> np.ndarray:
    return np.sum(np.square(counts - (cal_matrix @ x)))


class ReadoutMitigation(CountsAction):
    """Least-squares readout mitigator action."""
    def __init__(
        self,
        cal_matrix: np.ndarray,
        validate: bool = True
    ):
        super().__init__(validate=validate)
        self._cal_matrix = cal_matrix

    def _process(self, data: np.ndarray) -> np.ndarray:
        mitigated_counts = np.empty(data.size, dtype=object)
        for idx, counts_dict in enumerate(data):
            num_readout = len(list(counts_dict.keys())[0])
            num_states = 2 ** num_readout
            if self._validate and num_states != self._cal_matrix.shape[0]:
                raise ValueError('Readout mitigation matrix size mismatch')

            key_template = '{:0%db}' % num_readout

            counts_arr = np.array([counts_dict.get(key_template.format(i), 0)
                                   for i in range(num_states)], dtype=float)
            nshots = np.sum(counts_arr)

            constraints = {'type': 'eq', 'fun': lambda x: nshots - np.sum(x)} # pylint: disable=cell-var-from-loop
            bounds = sciopt.Bounds(0, nshots)
            res = sciopt.minimize(_objective, x0=counts_arr, args=(counts_arr, self._cal_matrix),
                                  method='SLSQP', constraints=constraints, bounds=bounds, tol=1.e-6)
            opt_counts = np.round(res.x).astype(int)
            mitigated_counts[idx] = {key_template.format(i): int(opt_counts[i])
                                     for i in range(num_states)}

        return mitigated_counts
