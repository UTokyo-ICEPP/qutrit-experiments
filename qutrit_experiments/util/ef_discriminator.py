r"""Identify the IQ-plane classifier boundary through numerical optimization.

Denote the GS tone width and measured IQ-plane data points for circuit :math:`j` as :math:`w_j`
and :math:`\vec{x}_{jk} = (i_{jk}, q_{jk})^T` (:math:`j = 0, \dots, N_{\text{circ}} - 1`,
:math:`k = 0, \dots, N_{\text{shot}} - 1`) respectively. The discriminator boundary is defined by
:math:`\vec{x} \cdot \vec{n} = d`, where the norm vector
:math:`\vec{n} := (\cos\theta, \sin\theta)^T` points towards the IQ region corresponding to the
classification result :math:`|2\rangle` and :math:`d` is the signed distance of the classification
boundary from the origin.

The classification result (:math:`|1\rangle \rightarrow 0` and :math:`|2\rangle \rightarrow 1`) for
each data point is approximated as

.. math::

    y_{jk} = \frac{1}{2} \mathrm{tanh}(s[\vec{x}_{jk} \cdot \vec{n} - d]) + \frac{1}{2}

where :math:`s` is a hyperparameter representing a scale factor. Then, for a given :math:`\theta`
and :math:`d`, the expectation value of :math:`Z_{12}` for circuit :math:`j` is

.. math::

    z_j = -\frac{1}{N_{\text{shot}}} \sum_{k} \mathrm{tanh}(s[\vec{x}_{jk} \cdot \vec{n} - d])

The value of :math:`z_j` should oscillate with respect to :math:`w_j` with the frequency and phase
offset already obtained from the OscillationAnalysis fit to the SVD-projected raw IQ data:

.. math::

    c_j = \cos (2 \pi f w_j + \phi)

An oscillation curve fit to :math:`z_j` amounts to minimizing

.. math::

    L(a, b; \theta, d) = \sum_j (a c_j + b - z_j)^2

for some amplitude :math:`a` and baseline offset :math:`b`. The best-fit parameters
:math:`\hat{a}(\theta, d)` and :math:`\hat{b}(\theta, d)` are given by solving

.. math::

    \frac{\partial L}{\partial a} = 0 \\
    \frac{\partial L}{\partial b} = 0,

which translates to

.. math::

    \begin{pmatrix} \sum_j c_j & N_{\mathrm{circ}} \\ \sum_j c_j^2 & \sum_j c_j \end{pmatrix}
    \begin{pmatrix} \hat{a} \\ \hat{b} \end{pmatrix}
    = \begin{pmatrix} \sum_j z_j \\ \sum_j c_j z_j \end{pmatrix}.

We seek to find a classification boundary that maximizes :math:`\hat{a}`.
"""
from typing import TYPE_CHECKING
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from sklearn.cluster import KMeans
from uncertainties import unumpy as unp
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.data_processing import DataProcessor

if TYPE_CHECKING:
    from qiskit_experiments.framework import ExperimentData

def fit_discriminator_boundary(
    wval: np.ndarray,
    samples: np.ndarray,
    freq: float,
    phase: float,
    idx_i1: int,
    maxiter: int,
    convergence: float
) -> tuple[float, float]:
    """Perform a fit to the experiment data.

    Args:
        wval: [N_circ]
        samples: [N_circ, N_shot, 1, 2]
        freq: Best-fit frequency.
        phase: Best-fit phase.
        idx_i1: The outermost index of the samples that should be the closest to i1.
        maxiter: Maximum number of SGD iterations. Set to 0 if K-means clusterization is sufficient.
        convergence: Loss convergence tolerance.
    """
    samples = np.squeeze(samples)

    # Get initial guesses for theta and dist from KMeans
    centers = KMeans(n_clusters=2, random_state=0).fit(samples.reshape(-1, 2)).cluster_centers_

    # Which center is closer to samples[0]?
    d = samples[idx_i1][:, None, :] - centers[None, :, :]
    distmean = np.mean(np.sqrt(np.sum(np.square(d), axis=-1)), axis=0)
    i1 = np.argmin(distmean)
    i2 = np.argmax(distmean)

    # The norm vector points from i1 to i2
    i1toi2 = centers[i2] - centers[i1]
    centroids_dist = np.sqrt(np.dot(i1toi2, i1toi2))

    vnorm = i1toi2 / centroids_dist
    theta = np.arctan2(vnorm[1], vnorm[0])

    dist = np.dot(vnorm, np.mean(centers, axis=0))

    if maxiter <= 0:
        return theta, dist

    # Because the data points go through SVD projection, sign of the y values are meaningless.
    # The target curve of the fit should correspond to <Z12>, so we adjust the phase to set
    # y[idx_i1] = cos(2pi * freq * wval[idx_i1] + phase) > 0.
    if np.cos(2. * np.pi * freq * wval[idx_i1] + phase) < 0.:
        phase += np.pi

    cval = np.cos(2. * np.pi * freq * wval + phase)
    sumc = np.sum(cval)
    cmat_inv_row0 = np.linalg.inv([[sumc, wval.shape[0]],
                                   [np.sum(np.square(cval)), sumc]])[0]

    @jax.jit
    def minus_ahat(params):
        nvec = jnp.array([jnp.cos(params[0]), jnp.sin(params[0])])
        zval = jnp.mean(jnp.tanh((jnp.dot(samples, nvec) - params[1]) / centroids_dist), axis=1)
        return -jnp.dot(cmat_inv_row0, jnp.array([jnp.sum(zval), jnp.sum(cval * zval)]))

    solver = jaxopt.BFGS(fun=minus_ahat, maxiter=maxiter, tol=convergence)
    res = solver.run(jnp.array([theta, dist]))
    theta, dist = map(float, res.params)

    while theta > np.pi:
        theta -= 2. * np.pi
    while theta < -np.pi:
        theta += 2. * np.pi

    return theta, dist


def ef_discriminator_analysis(
    experiment_data: 'ExperimentData',
    idx_i1: int,
    maxiter: int = 10000,
    convergence: float = 1.e-4
) -> tuple[float, float]:
    fit_results = next(res for res in experiment_data.analysis_results()
                       if res.name.startswith(PARAMS_ENTRY_PREFIX))
    popt = fit_results.value.params

    wval = np.array(list(d['metadata']['xval'] for d in experiment_data.data()))
    samples = unp.nominal_values(DataProcessor('memory')(experiment_data.data()))

    sort_indices = np.argsort(wval)
    wval = wval[sort_indices]
    samples = samples[sort_indices]

    return fit_discriminator_boundary(wval, samples, popt['freq'], popt['phase'], idx_i1,
                                      maxiter, convergence)
