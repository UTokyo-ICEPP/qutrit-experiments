"""Postprocessor to convert kerneled measurement results to counts."""

import numpy as np
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit_experiments.framework import ExperimentData


class LinearIQClassifierBase:
    """Base class for IQ classifier with a linear boundary."""
    def __init__(self, discriminators: list[tuple[float, float]]):
        self.discriminators = list(discriminators)

    def _classify(self, experiment_data: ExperimentData):
        # Number of discriminators must match the classical register width
        vnorms = np.array([[np.cos(theta), np.sin(theta)] for theta, _ in self.discriminators])
        dists = np.array([dist for _, dist in self.discriminators])

        # C: number of circuits, S: number of shots, Q: number of qubits
        memory = np.array([datum.pop('memory') for datum in experiment_data.data()])
        disc_res = np.asarray(np.einsum('csqi,qi->csq', memory, vnorms) - dists > 0.)
        packed = np.packbits(disc_res, axis=-1, bitorder='little') # [C, S, ceil(Q/8)]
        # packbits returns an array of uint8 -> do bitshift and sum to make full result bitstrings
        bin_res = np.sum(packed * np.power(2, np.arange(packed.shape[-1]) * 8), axis=-1) # [C, S]
        # sort to make counting easier
        bin_res.sort(axis=-1)

        key = '{:0%db}' % memory.shape[2]

        for datum, circ_res, circ_mem in zip(experiment_data.data(), bin_res, memory):
            counts = {}
            current = 0
            while current < circ_res.shape[0]:
                pos = np.searchsorted(circ_res, circ_res[current], side='right')
                counts[key.format(circ_res[current])] = pos - current
                current = pos

            datum['counts'] = Counts(counts)
            datum['memory_orig'] = circ_mem
            datum['meas_level'] = MeasLevel.CLASSIFIED
            if 'meas_return' in datum:
                datum.pop('meas_return')

    @classmethod
    def _update_metadata(cls, experiment_data: ExperimentData):
        experiment_data.metadata['meas_level'] = MeasLevel.CLASSIFIED
        try:
            experiment_data.metadata.pop('meas_return')
        except KeyError:
            pass

        for comp_metadata in experiment_data.metadata.get('component_metadata', []):
            cls._update_metadata(comp_metadata)

    def asarg(self):
        return ('iq_classification', self)


class SingleQutritLinearIQClassifier(LinearIQClassifierBase):
    """LinearIQClassifier for single-qutrit experiments."""
    # __name__ attribute needed for this object to be registered as an analysis callback (i.e.
    # postprocessor) to ExperimentData
    __name__ = 'iq_classification'

    def __init__(self, theta: float, dist: float):
        super().__init__([(theta, dist)])

    def __call__(self, experiment_data: ExperimentData):
        self._classify(experiment_data)
        self._update_metadata(experiment_data)


class ParallelSingleQutritLinearIQClassifier:
    """LinearIQClassifier for parallel single-qutrit experiments."""
    def __init__(self, discriminators: list[tuple[float, float]]):
        self.classifiers = [SingleQutritLinearIQClassifier(theta, phi)
                            for theta, phi in discriminators]

    def __call__(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata["component_child_index"]
        for ichild, classifier in zip(component_index, self.classifiers):
            classifier._classify(experiment_data.child_data(ichild))

        LinearIQClassifierBase._update_metadata(experiment_data)
