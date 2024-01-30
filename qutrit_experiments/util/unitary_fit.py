"""Functions for fitting a unitary to observation."""
from collections.abc import Sequence
from typing import Any, NamedTuple, Optional, Union
import jax
import jax.numpy as jnp
import jaxopt
import numpy as np
from uncertainties import unumpy as unp
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from .bloch import rescale_axis, so3_cartesian

axes = ['x', 'y', 'z']


def fit_unitary(
    data: list[dict[str, Any]],
    p0: Optional[Sequence[float]] = None,
    data_processor: Optional[DataProcessor] = None,
    plot: bool = True
) -> tuple[np.ndarray, NamedTuple, np.ndarray, np.ndarray, Union['matplotlib.figure.Figure', None]]:
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

    @jax.jit
    def objective(params):
        r_elements = so3_cartesian(params, npmod=jnp)[..., meas_bases, initial_states] * signs
        return jnp.sum(jnp.square(r_elements - unp.nominal_values(expvals)), axis=-1)

    solver = jaxopt.GradientDescent(fun=objective)

    fit_results = []
    for iax in range(3):
        for sign in [1., -1.]:
            p0 = np.zeros(3)
            p0[iax] = np.pi * sign / 2.
            fit_results.append(solver.run(p0))

    res = min(fit_results, key=lambda r: r.state.error)
    params = rescale_axis(np.array(res.params))
    expvals_pred = so3_cartesian(params)[..., meas_bases, initial_states] * signs

    if plot:
        ax = get_non_gui_ax()
        xvalues = np.arange(len(initial_states))
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel('Pauli expectation')
        ax.errorbar(xvalues, unp.nominal_values(expvals), unp.std_devs(expvals), fmt='o',
                    label='observed')
        ax.bar(xvalues, np.zeros_like(xvalues), 1., bottom=expvals_pred, fill=False,
               label='fit result')
        xticks = [f'{axes[basis]}|{axes[init]}{"+" if sign > 0 else "-"}'
                  for init, sign, basis in zip(initial_states, signs, meas_bases)]
        ax.set_xticks(xvalues, labels=xticks)
        ax.legend()
        figure = ax.get_figure()
    else:
        figure = None

    return params, res.state, expvals, expvals_pred, figure
