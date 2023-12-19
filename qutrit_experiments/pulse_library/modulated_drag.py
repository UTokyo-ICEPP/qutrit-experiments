"""Drag with modulation(s)."""

from typing import Optional, Union
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.library.symbolic_pulses import _PulseType, _lifted_gaussian, ScalableSymbolicPulse
from qiskit.utils import optionals as _optional
if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class ModulatedDrag(metaclass=_PulseType):
    """A Drag with modulation."""

    alias = 'ModulatedDrag'

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: ParameterValueType,
        sigma: ParameterValueType,
        beta: ParameterValueType,
        freq: Union[ParameterValueType, tuple[ParameterValueType, ...]],
        fractions: tuple[ParameterValueType, ...] = (),
        angle: Optional[ParameterValueType] = None,
        phases: Optional[tuple[ParameterValueType, ...]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> ScalableSymbolicPulse:
        """Create new pulse instance using modulated_gaussian_square().
        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            amp: The magnitude of the amplitude of the overall pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the function
                docstring for details.
            beta: The correction amplitude.
            freq: Modulation frequency(ies).
            fractions: Fractions of the magnitude of the amplitude assigned to each frequency
                component; see the function docstring for details.
            angle: The angle of the complex amplitude of the pulse. Default value 0.
            phases: Relative phases of additional frequency components; see the function docstring
                for details.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1. The default
                is ``True`` and the amplitude is constrained to 1.
        Returns:
            ScalableSymbolicPulse instance.
        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty.
        """
        return modulated_drag(cls.alias, duration, amp, sigma, beta, freq,
                              fractions=fractions, angle=angle, phases=phases,
                              name=name, limit_amplitude=limit_amplitude)


def modulated_drag(
    alias: str,
    duration: Union[int, ParameterExpression],
    amp: ParameterValueType,
    sigma: ParameterValueType,
    beta: ParameterValueType,
    freq: Union[ParameterValueType, tuple[ParameterValueType, ...]],
    fractions: tuple[ParameterValueType, ...] = (),
    angle: Optional[ParameterValueType] = None,
    phases: Optional[tuple[ParameterValueType, ...]] = None,
    name: Optional[str] = None,
    limit_amplitude: Optional[bool] = None,
) -> ScalableSymbolicPulse:
    """Function to directly return a ScalableSymbolicPulse instance.

    Args:
        alias: Pulse instance alias.
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the overall pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is.
        beta: The correction amplitude.
        freq: Modulation frequency(ies).
        fractions: Fractions of the magnitude of the amplitude assigned to each frequency
            component
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        phases: Relative phases of additional frequency components
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1. The default
            is ``True`` and the amplitude is constrained to 1.
    Returns:
        ScalableSymbolicPulse instance.

    If the ``freq`` parameter is a tuple, the fractions must also be a tuple with the number of
    entries identical or one less than that of ``freq``. In the latter case, the first frequency
    component is assigned the remainder of the amplitude. If ``fractions`` has the same number of
    entries as ``freq``, the fraction values do not have to add up to 1 (will be normalized). The
    relative phases (``phases``), if given, must be a tuple with one fewer entries than ``freq``.
    The ``angle`` parameter will be the phase of the first frequency component, and ``phases`` will
    set the phase of the remaining components.
    """
    if not isinstance(freq, tuple):
        freq = (freq,)

    ncomp = len(freq)

    if len(fractions) == ncomp:
        norm = sum(fractions)
        if norm == 0.:
            raise ValueError('Invalid fractions adding up to zero')
        if norm != 1.:
            fractions = tuple(f / norm for f in fractions)
    elif len(fractions) == ncomp - 1:
        fractions = (1. - sum(fractions, 0.),) + fractions
    else:
        raise ValueError('Fractions array has invalid length')
    for f in fractions:
        try:
            if f < 0. or f > 1.:
                raise ValueError(f'Invalid fractions value {f}')
        except TypeError: # Parameter
            pass

    if phases is not None and len(phases) != ncomp - 1:
        raise ValueError('Phases must have length ncomp - 1')

    parameters = {'sigma': sigma, 'beta': beta}
    if ncomp == 1:
        parameters['freq'] = freq[0]
    else:
        for icomp in range(ncomp):
            parameters[f'freq{icomp}'] = freq[icomp]
            parameters[f'fraction{icomp}'] = fractions[icomp]
            if phases is not None and icomp != 0:
                parameters[f'phase{icomp}'] = phases[icomp - 1]

    # Prepare symbolic expressions
    _t, _duration, _amp, _sigma, _beta, _angle = sym.symbols(
        't, duration, amp, sigma, beta, angle'
    )
    twopit = 2 * sym.pi * _t
    if ncomp == 1:
        _modulation = sym.exp(sym.I * twopit * sym.Symbol('freq'))
    else:
        _modulation = sym.Symbol('fraction0') * sym.exp(sym.I * twopit * sym.Symbol('freq0'))
        for icomp in range(1, ncomp):
            _modulation += (sym.Symbol(f'fraction{icomp}')
                            * sym.exp(sym.I * (twopit * sym.Symbol(f'freq{icomp}')
                                               + sym.Symbol(f'phase{icomp}'))))

    _center = _duration / 2

    _gauss = _lifted_gaussian(_t, _center, _duration + 1, _sigma)
    _deriv = -(_t - _center) / (_sigma**2) * _gauss

    envelope_expr = _amp * sym.exp(sym.I * _angle) * _modulation * (_gauss + sym.I * _beta * _deriv)

    consts_expr = _sigma > 0
    valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

    instance = ScalableSymbolicPulse(
        pulse_type=alias,
        duration=duration,
        amp=amp,
        angle=angle,
        parameters=parameters,
        name=name,
        limit_amplitude=limit_amplitude,
        envelope=envelope_expr,
        constraints=consts_expr,
        valid_amp_conditions=valid_amp_conditions_expr,
    )
    instance.validate_parameters()

    return instance
