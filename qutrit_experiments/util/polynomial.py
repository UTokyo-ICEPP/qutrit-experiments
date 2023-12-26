from collections.abc import Callable, Sequence
from typing import Optional, Union
import numpy as np
import numpy.polynomial as poly

def sparse_poly_fitfunc(powers: Sequence[int]) -> Callable:
    """Return a sparse polynomial function to be used with scipy.optimize.curve_fit."""
    powers = np.sort(powers)
    full_coeffs = np.zeros(max(powers) + 1)

    def fitfunc(x, *coeffs):
        full_coeffs[powers] = coeffs
        return poly.polynomial.polyval(x, full_coeffs)

    return fitfunc


class PolynomialOrder:
    def __init__(
        self,
        order: Union[int, Sequence[int], 'PolynomialOrder'],
        int_constraint: Optional[str] = None
    ):
        if isinstance(order, int):
            if int_constraint == 'even':
                self.powers = list(range(0, order + 1, 2))
            elif int_constraint == 'odd':
                self.powers = list(range(1, order + 1, 2))
            else:
                self.powers = list(range(order + 1))
        elif isinstance(order, PolynomialOrder):
            self.powers = list(order.powers)
        else:
            self.powers = list(order)

    @property
    def order(self):
        return max(self.powers)
