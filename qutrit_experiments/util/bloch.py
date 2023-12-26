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


def amp_matrix(psi: array_like, phi: array_like) -> array_like:
    """Compute the amplitude of the projection of the Bloch trajectory."""
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    A = spsi * np.sqrt(((cpsi * cphi) ** 2) + (sphi ** 2))
    B = spsi * np.sqrt(((cpsi * sphi) ** 2) + (cphi ** 2))
    C = np.sqrt((spsi ** 4) * (sphi ** 2) * (cphi ** 2) + (cpsi ** 2))

    return np.array([
        [1. - ((spsi * cphi) ** 2), C, A],
        [C, 1. - ((spsi * sphi) ** 2), B],
        [A, B, spsi ** 2 * np.ones_like(phi)]
    ])

def amp_func(basis: array_like, psi: array_like, phi: array_like) -> array_like:
    """Get the amplitude matrix element."""
    values = amp_matrix(psi, phi)
    basis = np.asarray(basis, dtype=int)
    init = basis // 3
    meas = basis % 3
    return values[meas, init]

def phase_matrix(psi: array_like, phi: array_like, delta: array_like) -> array_like:
    """Compute the phase offset of the projection of the Bloch trajectory."""
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    alpha = np.arctan2(-sphi, -cpsi * cphi)
    beta = np.arctan2(cphi, -cpsi * sphi)
    gamma = np.arctan2(-cpsi, -(spsi ** 2) * sphi * cphi)

    return np.array([
        [delta, -gamma + delta, alpha + delta],
        [gamma + delta, delta, beta + delta],
        [-alpha + delta, -beta + delta, delta]
    ])

def phase_func(basis: array_like, psi: array_like, phi: array_like, delta: array_like) -> array_like:
    """Get the phase matrix element."""
    values = phase_matrix(psi, phi, delta)
    basis = np.asarray(basis, dtype=int)
    init = basis // 3
    meas = basis % 3
    return values[meas, init]

def base_matrix(psi: array_like, phi: array_like) -> array_like:
    """Compute the center of oscillation of the projection of the Bloch trajectory."""
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    spsi2 = np.square(spsi)
    spsicpsi = spsi * cpsi

    return np.array([
        [spsi2 * (cphi ** 2), spsi2 * sphi * cphi, spsicpsi * cphi],
        [spsi2 * sphi * cphi, spsi2 * (sphi ** 2), spsicpsi * sphi],
        [spsicpsi * cphi, spsicpsi * sphi, cpsi ** 2 * np.ones_like(phi)]
    ])

def base_func(basis: array_like, psi: array_like, phi: array_like) -> array_like:
    """Get the base matrix element."""
    values = base_matrix(psi, phi)
    basis = np.asarray(basis, dtype=int)
    init = basis // 3
    meas = basis % 3
    return values[meas, init]
