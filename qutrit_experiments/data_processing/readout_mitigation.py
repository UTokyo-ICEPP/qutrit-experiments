"""Least-squares readout mitigator action."""
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np
import scipy.optimize as sciopt
from qiskit.result import BaseReadoutMitigator, CorrelatedReadoutMitigator, LocalReadoutMitigator
from qiskit_experiments.data_processing.nodes import CountsAction


def _objective(x: np.ndarray, counts: np.ndarray, cal_matrix: np.ndarray) -> np.ndarray:
    return np.sum(np.square(counts - (cal_matrix @ x)))


class ReadoutMitigation(CountsAction):
    """Least-squares readout mitigator action.

    Args:
        cal_matrix: A [2**n, 2**n] array or a list of n [2, 2] arrays. In the former case, the
            least significant digit corresponds to physical_qubits[0]. In the latter case, the first
            matrix is for physical_qubits[0].
        readout_mitigator: Alternative construction of the node is through a readout mitigator and
            a list of qubits.
        physical_qubits: The list of physical qubits if constructing the node from a readout
            mitigator.
    """
    def __init__(
        self,
        cal_matrix: Optional[Union[np.ndarray, list[np.ndarray]]] = None,
        readout_mitigator: Optional[BaseReadoutMitigator] = None,
        physical_qubits: Optional[Sequence[int]] = None,
        validate: bool = True
    ):
        super().__init__(validate=validate)
        if isinstance(readout_mitigator, LocalReadoutMitigator):
            cal_matrix = [readout_mitigator.assignment_matrix(qubit) for qubit in physical_qubits]
        elif isinstance(readout_mitigator, CorrelatedReadoutMitigator):
            cal_matrix = readout_mitigator.assignment_matrix(physical_qubits)
            # CorrelatedReadoutMitigator.assignment_matrix() implicitly sorts the qubits through the
            # use of set, so we reorder the axes. Also note that unused qubits are assumed to be at
            # |0> state.
            indices = list(reversed([readout_mitigator._qubit_index(iq) for iq in physical_qubits]))
            sorted_indices = list(reversed(sorted(indices)))
            if indices != sorted_indices:
                nq = len(indices)
                transpose = [sorted_indices.index(i) for i in indices] # row reordering
                transpose += [t + nq for t in transpose] # column reordering
                cal_matrix = cal_matrix.reshape((2,) * (2 * nq))
                cal_matrix = cal_matrix.transpose(transpose)
                cal_matrix = cal_matrix.reshape((2 ** nq,) * 2)

        if cal_matrix is None:
            raise RuntimeError('Assignment matrix not given')

        if isinstance(cal_matrix, list):
            self._cal_matrix = np.array(1)
            for mat in cal_matrix:
                self._cal_matrix = np.kron(mat, self._cal_matrix)
        else:
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
