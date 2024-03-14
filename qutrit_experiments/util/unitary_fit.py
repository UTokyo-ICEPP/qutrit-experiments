"""Functions for fitting a unitary to observation."""
import logging
from threading import Lock
from typing import Any, NamedTuple, Optional
from matplotlib.figure import Figure
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from numpy.linalg import LinAlgError
from uncertainties import correlated_values, ufloat, unumpy as unp
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from .bloch import rescale_axis, so3_cartesian

logger = logging.getLogger(__name__)
axes = ['x', 'y', 'z']
twopi = 2. * np.pi
default_maxiter = 10000
default_tol = 1.e-4


def fit_unitary(
    data: list[dict[str, Any]],
    data_processor: Optional[DataProcessor] = None,
    maxiter: int = default_maxiter,
    tol: float = default_tol
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
    maxiter: int = default_maxiter,
    tol: float = default_tol
) -> tuple[np.ndarray, NamedTuple]:
    logger.debug('Fit unitary to expval %d datapoints maxiter %s tol %s', len(expvals), maxiter,
                 tol)
    if signs is None:
        signs = np.ones_like(initial_states)

    p0s = []
    for iax in range(3):
        for sign in [1., -1.]:
            p0 = np.zeros(3)
            p0[iax] = np.pi * sign / 2.
            p0s.append(p0)

    fitter = UnitaryFitObjective(expvals.shape[0], maxiter, tol)
    objective_args = (meas_bases, initial_states, signs, unp.nominal_values(expvals),
                      unp.std_devs(expvals))
    fit_result = fitter.vsolve(jnp.array(p0s), *objective_args)
    fvals = fitter.vobj(fit_result.params, *objective_args)
    iopt = np.argmin(fvals)
    # Renormalize the rotation parameters so that the norm fits within [0, pi].
    popt = rescale_axis(fit_result.params[iopt])

    try:
        pcov = np.linalg.inv(fitter.hess(popt, *objective_args) * 0.5)
        popt_ufloats = correlated_values(nom_values=popt, covariance_mat=pcov, tags=['x', 'y', 'z'])
    except LinAlgError:
        logger.warning('Invalid covariance encountered. Setting paramater uncertainties to inf')
        popt_ufloats = tuple(ufloat(p, np.inf) for p in popt)

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
    ax.axhline(0., color='black', linestyle='--')
    ec = ax.errorbar(xvalues, unp.nominal_values(expvals), unp.std_devs(expvals), fmt='o',
                        label='observed')
    ax.bar(xvalues, np.zeros_like(xvalues), 1., bottom=expvals_pred, fill=False,
           edgecolor=ec.lines[0].get_markerfacecolor(), label='fit result')
    xticks = [f'{axes[basis]}|{axes[init]}{"+" if sign > 0 else "-"}'
                for init, sign, basis in zip(initial_states, signs, meas_bases)]
    ax.set_xticks(xvalues, labels=xticks)
    ax.legend()
    return ax.get_figure()


class UnitaryFitObjective:
    """Objective function and derivatives for unitary fit."""
    _lock = Lock()
    _instances = {}
    
    @staticmethod
    def fun(params, meas_bases, initial_states, signs, expvals, expvals_err):
        r_elements = so3_cartesian(params, npmod=jnp)[..., meas_bases, initial_states] * signs
        return jnp.sum(
            jnp.square((r_elements - expvals) / expvals_err),
            axis=-1
        )

    def __new__(cls, num_args: int = 9, maxiter: int = default_maxiter, tol: float = default_tol):
        key = (num_args, maxiter, tol)
        with cls._lock:
            if (instance := cls._instances.get(key)) is None:
                instance = super().__new__(cls)
                instance._initialize(num_args, maxiter, tol)
                cls._instances[key] = instance

        return instance
    
    def __init__(self, num_args: int = 9, maxiter: int = default_maxiter, tol: float = default_tol):
        pass

    def _initialize(self, num_args: int, maxiter: int, tol: float):
        ints = np.zeros(num_args, dtype=np.int64)
        floats = np.zeros(num_args, dtype=np.float64)
        args = (ints, ints, ints, floats, floats)
        in_axes = [0] + ([None] * len(args))
        self.vobj = jax.jit(
            jax.vmap(self.fun, in_axes=in_axes)
        ).lower(np.zeros((6, 3)), *args).compile()
        self.vsolve = jax.jit(
            jax.vmap(jaxopt.GradientDescent(fun=self.fun, maxiter=maxiter, tol=tol).run,
                     in_axes=in_axes)
        ).lower(np.zeros((6, 3)), *args).compile()
        self.hess = jax.jit(
            jax.hessian(self.fun)
        ).lower(jnp.zeros(3), *args).compile()
