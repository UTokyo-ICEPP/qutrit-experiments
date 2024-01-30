"""Functions for fitting a unitary to observation."""
from collections.abc import Sequence
from typing import Any, Optional
import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
import numpy as np
from uncertainties import unumpy as unp
from qiskit_experiments.data_processing import BasisExpectationValue, DataProcessor, Probability
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from .bloch import rescale_axis, so3_cartesian

axes = ['x', 'y', 'z']
qpt_prep_states = [('z', 1), ('z', -1), ('x', 1), ('y', 1)]
qpt_meas_bases = ['z', 'x', 'y']
hilbert_states = np.array([
    np.array([1., 1.]) / np.sqrt(2.),
    np.array([1., -1.]) / np.sqrt(2.),
    np.array([1., 1.j]) / np.sqrt(2.),
    np.array([1.j, 1.]) / np.sqrt(2.),
    np.array([1., 0.]),
    np.array([0., 1.])
], dtype=complex)


def fit_unitary(
    data: list[dict[str, Any]],
    p0: Optional[Sequence[float]] = None,
    data_processor: Optional[DataProcessor] = None,
    plot: bool = True
):
    if data_processor is None:
        data_processor = DataProcessor('counts', [Probability('1'), BasisExpectationValue()])

    expvals = data_processor(data)
    initial_states = []
    signs = []
    meas_bases = []
    for datum in data:
        metadata = datum['metadata']
        if (p_idx := metadata.get('p_idx')) is not None:
            state, sign = qpt_prep_states[p_idx[0]]
        else:
            state = metadata['initial_state']
            sign = metadata.get('initial_state_sign', 1)
        if (m_idx := metadata.get('m_idx')) is not None:
            basis = qpt_meas_bases[m_idx[0]]
        else:
            basis = metadata['meas_basis']
        initial_states.append(axes.index(state))
        signs.append(sign)
        meas_bases.append(axes.index(basis))
    initial_states = np.array(initial_states, dtype=int)
    signs = np.array(signs, dtype=int)
    meas_bases = np.array(meas_bases, dtype=int)

    @jax.jit
    def objective(params):
        r_elements = so3_cartesian(params, npmod=jnp)[..., initial_states, meas_bases] * signs
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
    expvals_pred = so3_cartesian(params)[..., initial_states, meas_bases] * signs

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

    return params, res.state, expvals_pred, figure
