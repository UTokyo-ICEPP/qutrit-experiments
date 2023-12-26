"""Linear discriminator in the IQ plane."""
from typing import Any
import numpy as np
from qiskit_experiments.data_processing import BaseDiscriminator


class LinearDiscriminator(BaseDiscriminator):
    """Linear discriminator in the IQ plane."""
    def __init__(self, theta, dist):
        self.theta = theta
        self.vnorm = np.array([np.cos(theta), np.sin(theta)])
        self.dist = dist

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.asarray(np.dot(data, self.vnorm) - self.dist > 0.).astype(int).astype(str)

    def config(self) -> dict[str, Any]:
        return {'theta': self.theta, 'dist': self.dist}

    def is_trained(self) -> bool:
        return True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> 'LinearDiscriminator':
        return cls(**config)
