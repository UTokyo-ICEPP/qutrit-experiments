"""Functions for Bloch-sphere analyses."""
from collections.abc import Sequence
from numbers import Number
from typing import Union
import jax.numpy as jnp
import numpy as np
from uncertainties import unumpy as unp

array_like = Union[Number, np.ndarray, Sequence[Number]]
twopi = 2. * np.pi
# Define math functions not in unp
for fname in ['array', 'square', 'sum', 'einsum', 'moveaxis']:
    setattr(unp, fname, getattr(np, fname))


def unit_bound(x: array_like) -> array_like:
    return np.minimum(np.maximum(x, -1.), 1.)


def pos_unit_bound(x: array_like) -> array_like:
    return np.minimum(np.maximum(x, 0.), 1.)


def so3_polar(theta: array_like, psi: array_like, phi: array_like, npmod=np) -> array_like:
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


def su2_cartesian(xyz: array_like, npmod=np):
    axis, norm = normalized_rotation_axis(xyz, npmod=npmod)
    return su2_cartesian_axnorm(axis, norm, npmod=npmod)


def su2_cartesian_axnorm(axis: array_like, norm: array_like, npmod=np):
    sigma = npmod.sum(axis[..., None, None] * paulis, axis=-3)
    unitary = npmod.cos(norm / 2.)[..., None, None] * np.eye(2, dtype='complex128')
    unitary -= 1.j * npmod.sin(norm / 2.)[..., None, None] * sigma
    return unitary


def su2_cartesian_params(unitary: array_like, npmod=np):
    if npmod is np:
        unitary = np.asarray(unitary)
    sin_axis = -npmod.einsum('...ij,kji->...k', unitary, paulis).imag / 2.
    theta = npmod.arccos(npmod.einsum('...ii->...', unitary).real / 2.)[..., None] * 2.
    return sin_axis / npmod.sin(theta / 2.) * theta


def so3_cartesian(xyz: array_like, npmod=np):
    rot_axis, norm = normalized_rotation_axis(xyz, npmod=npmod)
    ctheta = npmod.cos(norm)
    ictheta = 1. - ctheta
    stheta = npmod.sin(norm)
    x, y, z = npmod.moveaxis(rot_axis, -1, 0)
    x2, y2, z2 = npmod.moveaxis(npmod.square(rot_axis), -1, 0)
    return npmod.array([
        [ctheta + x2 * ictheta, x * y * ictheta - z * stheta, x * z * ictheta + y * stheta],
        [y * x * ictheta + z * stheta, ctheta + y2 * ictheta, y * z * ictheta - x * stheta],
        [z * x * ictheta - y * stheta, z * y * ictheta + x * stheta, ctheta + z2 * ictheta]
    ])


def so3_cartesian_params(matrix: array_like, npmod=np):
    if npmod is np:
        matrix = np.asarray(matrix)
    sin_axis = np.moveaxis(
        npmod.array([
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1]
        ]) / 2.,
        0, -1
    )
    abs_sin = npmod.sqrt(npmod.sum(npmod.square(sin_axis), axis=-1))
    # Defining the axis to have the orientation where θ∊[0, π]
    axis = sin_axis / abs_sin[..., None]
    # Sum of offdiagonals = 2(uxuy+uyuz+uzux)(1-cosθ)
    cos = 1. - (npmod.sum(matrix[..., [2, 1, 0, 2, 1, 0], [1, 2, 2, 0, 0, 1]], axis=-1)
                / 2. / npmod.sum(axis[..., [0, 1, 2]] * axis[..., [1, 2, 0]], axis=-1))
    theta = npmod.arccos(cos)
    return axis * theta


def normalized_rotation_axis(xyz: array_like, npmod=np):
    if npmod is np:
        xyz = np.asarray(xyz)
    norm = npmod.sqrt(npmod.sum(npmod.square(xyz), axis=-1))
    if npmod is unp:
        rot_axis = xyz / np.where(unp.nominal_values(norm) == 0., 1., norm)[..., None]
    else:
        # This procedure and another "where" below cures the gradient becoming nan at the origin but
        # at the cost of it becoming zero (where the true limit is nonzero but depends on the direction
        # of approach)
        norm = npmod.where(npmod.isclose(norm, 0.), 0., norm)
        rot_axis = xyz / npmod.where(norm == 0., 1., norm)[..., None]
    return rot_axis, norm


def rescale_axis(xyz: array_like):
    """Rescale the axis vector so its norm is less than pi."""
    # Take a copy of the input because we'll be rescaling it
    xyz = np.array(xyz)
    if (ndim := xyz.ndim) == 1:
        xyz = xyz[None, :]

    norm = np.sqrt(np.sum(np.square(xyz), axis=-1))
    large_norm = norm > twopi
    xyz[large_norm] *= 1. % (twopi / norm[large_norm][..., None])

    norm = np.sqrt(np.sum(np.square(xyz), axis=-1))
    large_norm = norm > np.pi
    xyz[large_norm] *= (1. - twopi / norm[large_norm][..., None])
    if ndim == 1:
        return xyz[0]
    return xyz
