"""Functions for fitting a unitary to observation."""
import logging
from typing import Any, NamedTuple, Optional, Union
from matplotlib.figure import Figure
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from uncertainties import correlated_values, unumpy as unp
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from .bloch import rescale_axis, so3_cartesian

logger = logging.getLogger(__name__)
axes = ['x', 'y', 'z']
twopi = 2. * np.pi


def fit_unitary(
    data: list[dict[str, Any]],
    data_processor: Optional[DataProcessor] = None,
    maxiter: Optional[int] = None,
    tol: Optional[float] = None
) -> tuple[np.ndarray, NamedTuple, np.ndarray, np.ndarray, tuple]:
    expvals, initial_states, meas_bases, signs = extract_input_values(data, data_processor)
    popt_ufloats, state = fit_unitary_to_expval(expvals, initial_states, meas_bases, signs=signs,
                                                maxiter=maxiter, tol=tol)
    expvals_pred = so3_cartesian(unp.nominal_values(popt_ufloats))[..., meas_bases, initial_states]
    expvals_pred *= signs
    return popt_ufloats, state, expvals_pred, (expvals, initial_states, meas_bases, signs)


def extract_input_values(
    data: list[dict[str, Any]],
    data_processor: Optional[DataProcessor] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if data_processor is None:
        data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])

    expvals = data_processor(data)
    initial_states = []
    signs = []
    meas_bases = []
    for datum in data:
        metadata = datum['metadata']
        initial_states.append(axes.index(metadata['initial_state']))
        signs.append(metadata.get('initial_state_sign', 1))
        meas_bases.append(axes.index(metadata['meas_basis']))
    initial_states = np.array(initial_states, dtype=int)
    signs = np.array(signs, dtype=int)
    meas_bases = np.array(meas_bases, dtype=int)

    return expvals, initial_states, meas_bases, signs


def fit_unitary_to_expval(
    expvals: np.ndarray,
    initial_states: np.ndarray,
    meas_bases: np.ndarray,
    signs: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    tol: Optional[float] = None
) -> tuple[np.ndarray, NamedTuple]:
    logger.debug('Fit unitary to expval %d datapoints maxiter %s tol %s', len(expvals), maxiter,
                 tol)
    if signs is None:
        signs = np.ones_like(initial_states)

    @jax.jit
    def objective(params):
        r_elements = so3_cartesian(params, npmod=jnp)[..., meas_bases, initial_states] * signs
        return jnp.sum(
            jnp.square((r_elements - unp.nominal_values(expvals)) / unp.std_devs(expvals)),
            axis=-1
        )

    options = {}
    if maxiter:
        options['maxiter'] = maxiter
    if tol:
        options['tol'] = tol
    solver = jaxopt.GradientDescent(fun=objective, **options)

    p0s = []
    for iax in range(3):
        for sign in [1., -1.]:
            p0 = np.zeros(3)
            p0[iax] = np.pi * sign / 2.
            p0s.append(p0)

    fit_result = jax.vmap(solver.run)(jnp.array(p0s))
    fvals = jax.vmap(objective)(fit_result.params)
    iopt = np.argmin(fvals)
    # Renormalize the rotation parameters so that the norm fits within [0, pi].
    popt = rescale_axis(fit_result.params[iopt])

    pcov = np.linalg.inv(jax.hessian(objective)(popt) * 0.5)
    popt_ufloats = correlated_values(nom_values=popt, covariance_mat=pcov, tags=['x', 'y', 'z'])

    # state is a namedtuple
    values = []
    for sval in fit_result.state:
        if sval is None:
            values.append(None)
        else:
            values.append(np.array(sval[iopt]))
    state = fit_result.state.__class__(*values)

    return np.array(popt_ufloats), state


def plot_unitary_fit(
    popt_ufloats: np.ndarray,
    expvals: np.ndarray,
    initial_states: np.ndarray,
    meas_bases: np.ndarray,
    signs: Optional[np.ndarray] = None
) -> Figure:
    expvals_pred = so3_cartesian(unp.nominal_values(popt_ufloats))[..., meas_bases, initial_states]
    if signs is not None:
        expvals_pred *= signs

    ax = get_non_gui_ax()
    xvalues = np.arange(len(initial_states))
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel('Pauli expectation')
    ec = ax.errorbar(xvalues, unp.nominal_values(expvals), unp.std_devs(expvals), fmt='o',
                        label='observed')
    ax.bar(xvalues, np.zeros_like(xvalues), 1., bottom=expvals_pred, fill=False,
            edgecolor=ec.lines[0].get_markerfacecolor(), label='fit result')
    xticks = [f'{axes[basis]}|{axes[init]}{"+" if sign > 0 else "-"}'
                for init, sign, basis in zip(initial_states, signs, meas_bases)]
    ax.set_xticks(xvalues, labels=xticks)
    ax.legend()
    return ax.get_figure()
