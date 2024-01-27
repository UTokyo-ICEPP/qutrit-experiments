"""Functions for Bloch-sphere analyses."""
from collections.abc import Sequence
from numbers import Number
from typing import Union
import numpy as np

array_like = Union[Number, np.ndarray, Sequence[Number]]


def unit_bound(x: array_like) -> array_like:
    return np.minimum(np.maximum(x, -1.), 1.)


def pos_unit_bound(x: array_like) -> array_like:
    return np.minimum(np.maximum(x, 0.), 1.)


def rotation_matrix(theta: array_like, psi: array_like, phi: array_like, npmod=np) -> array_like:
    cpsi = npmod.cos(psi)
    spsi = npmod.sin(psi)
    cphi = npmod.cos(phi)
    sphi = npmod.sin(phi)

    A = spsi * npmod.sqrt(((cpsi * cphi) ** 2) + (sphi ** 2))
    B = spsi * npmod.sqrt(((cpsi * sphi) ** 2) + (cphi ** 2))
    C = npmod.sqrt((spsi ** 4) * (sphi ** 2) * (cphi ** 2) + (cpsi ** 2))
    alpha = npmod.arctan2(-sphi, -cpsi * cphi)
    beta = npmod.arctan2(cphi, -cpsi * sphi)
    gamma = npmod.arctan2(-cpsi, -(spsi ** 2) * sphi * cphi)
    spsi2 = npmod.square(spsi)
    spsicpsi = spsi * cpsi

    amp = npmod.array([
        [1. - ((spsi * cphi) ** 2), C, A],
        [C, 1. - ((spsi * sphi) ** 2), B],
        [A, B, spsi ** 2 * npmod.ones_like(phi)]
    ])
    phase = npmod.array([
        [0., -gamma, alpha],
        [gamma, 0., beta],
        [-alpha, -beta, 0.]
    ])
    base = npmod.array([
        [spsi2 * (cphi ** 2), spsi2 * sphi * cphi, spsicpsi * cphi],
        [spsi2 * sphi * cphi, spsi2 * (sphi ** 2), spsicpsi * sphi],
        [spsicpsi * cphi, spsicpsi * sphi, cpsi ** 2 * npmod.ones_like(phi)]
    ])

    if npmod is np:
        theta = np.asarray(theta)
    theta_dims = tuple(range(theta.ndim))
    amp = npmod.expand_dims(amp, theta_dims)
    phase = npmod.expand_dims(phase, theta_dims)
    base = npmod.expand_dims(base, theta_dims)
    theta = npmod.expand_dims(theta, (-2, -1))
    return amp * npmod.cos(theta + phase) + base


paulis = np.array([
    [[0., 1.], [1., 0.]],
    [[0., -1.j], [1.j, 0.]],
    [[1., 0.], [0., -1.]]
], dtype='complex128')

def rotation_matrix_xyz(xyz: array_like, npmod=np):
    if npmod is np:
        xyz = np.asarray(xyz)
    norm = npmod.sqrt(npmod.sum(npmod.square(xyz), axis=-1))
    sigma = npmod.sum((xyz / npmod.expand_dims(norm, -1))[..., None, None] * paulis, axis=-3)
    unitary = npmod.cos(norm / 2.)[..., None, None] * npmod.eye(2, dtype='complex128')
    unitary -= 1.j * npmod.sin(norm / 2.)[..., None, None] * sigma
    return unitary
