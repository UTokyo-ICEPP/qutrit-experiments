from numbers import Number
from typing import Optional, Union

from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse import PulseError
from qiskit.utils import optionals as _optional
if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym

from qiskit.pulse.library.symbolic_pulses import _PulseType, _lifted_gaussian, ScalableSymbolicPulse


class ModulatedGaussianSquare(metaclass=_PulseType):
    """A GaussianSquare with modulation."""

    alias = 'ModulatedGaussianSquare'

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        amp: ParameterValueType,
        sigma: ParameterValueType,
        freq: Union[ParameterValueType, tuple[ParameterValueType, ...]],
        fractions: tuple[ParameterValueType, ...] = (),
        width: Optional[ParameterValueType] = None,
        angle: Optional[ParameterValueType] = None,
        risefall_sigma_ratio: Optional[ParameterValueType] = None,
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
            freq: Modulation frequency(ies).
            fractions: Fractions of the magnitude of the amplitude assigned to each frequency
                component; see the function docstring for details.
            width: The duration of the embedded square pulse.
            angle: The angle of the complex amplitude of the pulse. Default value 0.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
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
        return modulated_gaussiansquare(cls.alias, duration, amp, sigma, freq,
                                        fractions=fractions, width=width, angle=angle,
                                        risefall_sigma_ratio=risefall_sigma_ratio, phases=phases,
                                        name=name, limit_amplitude=limit_amplitude)


def modulated_gaussiansquare(
    alias: str,
    duration: Union[int, ParameterExpression],
    amp: ParameterValueType,
    sigma: ParameterValueType,
    freq: Union[ParameterValueType, tuple[ParameterValueType, ...]],
    fractions: tuple[ParameterValueType, ...] = (),
    width: Optional[ParameterValueType] = None,
    angle: Optional[ParameterValueType] = None,
    risefall_sigma_ratio: Optional[ParameterValueType] = None,
    phases: Optional[tuple[ParameterValueType, ...]] = None,
    name: Optional[str] = None,
    limit_amplitude: Optional[bool] = None,
) -> ScalableSymbolicPulse:
    """Function to directly return a ScalableSymbolicPulse instance.

    Args:
        alias: Pulse instance alias.
        duration: Pulse length in terms of the sampling period `dt`.
        amp: The magnitude of the amplitude of the overall pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is
        freq: Modulation frequency(ies).
        fractions: Fractions of the magnitude of the amplitude assigned to each frequency
            component
        width: The duration of the embedded square pulse.
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        risefall_sigma_ratio: The ratio of each risefall duration to sigma.
        phases: Relative phases of additional frequency components
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1. The default
            is ``True`` and the amplitude is constrained to 1.
    Returns:
        ScalableSymbolicPulse instance.
    Raises:
        PulseError: When width and risefall_sigma_ratio are both empty.

    If the ``freq`` parameter is a tuple, the fractions must also be a tuple with the number of
    entries identical or one less than that of ``freq``. In the latter case, the first frequency
    component is assigned the remainder of the amplitude. If ``fractions`` has the same number of
    entries as ``freq``, the fraction values do not have to add up to 1 (will be normalized). The
    relative phases (``phases``), if given, must be a tuple with one fewer entries than ``freq``.
    The ``angle`` parameter will be the phase of the first frequency component, and ``phases`` will
    set the phase of the remaining components.
    """
    # Convert risefall_sigma_ratio into width which is defined in OpenPulse spec
    if width is None:
        if risefall_sigma_ratio is None:
            raise PulseError(
                'Either the pulse width or the risefall_sigma_ratio parameter must be specified.'
            )
        else:
            width = duration - 2.0 * risefall_sigma_ratio * sigma
            support = duration
    else:
        if risefall_sigma_ratio is None:
            support = duration
        else:
            support = width + 2. * risefall_sigma_ratio * sigma

    # Normalize the frequency, fraction, and phase parameters
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
        raise ValueError('Fractions array have invalid length')

    if phases is not None and len(phases) != ncomp - 1:
        raise ValueError('phases must have length ncomp - 1')

    parameters = {'sigma': sigma, 'width': width, 'support': support}
    if ncomp == 1:
        parameters['freq'] = freq[0]
    else:
        for icomp in range(ncomp):
            parameters[f'freq{icomp}'] = freq[icomp]
            parameters[f'fraction{icomp}'] = fractions[icomp]
            if phases is not None and icomp != 0:
                parameters[f'phase{icomp}'] = phases[icomp - 1]

    # Prepare symbolic expressions
    _t, _duration, _amp, _sigma, _width, _support, _angle = sym.symbols(
        't, duration, amp, sigma, width, support, angle'
    )
    twopit = 2 * sym.pi * _t
    if ncomp == 1:
        _modulation = sym.exp(sym.I * twopit * sym.Symbol('freq'))
    else:
        _fractions = [sym.Symbol(f'fraction{icomp}') for icomp in range(ncomp)]
        _modulation = _fractions[0] * sym.exp(sym.I * twopit * sym.Symbol('freq0'))
        for icomp in range(1, ncomp):
            _modulation += (_fractions[icomp] * sym.exp(sym.I * (
                twopit * sym.Symbol(f'freq{icomp}') + sym.Symbol(f'phase{icomp}')
            )))

    _center = _support / 2

    _sq_t0 = _center - _width / 2
    _sq_t1 = _center + _width / 2

    _gaussian_ledge = _lifted_gaussian(_t, _sq_t0, -1, _sigma)
    _gaussian_redge = _lifted_gaussian(_t, _sq_t1, _support + 1, _sigma)

    envelope_expr = (
        _amp * sym.exp(sym.I * _angle) * _modulation
        * sym.Piecewise(
            (_gaussian_ledge, _t <= _sq_t0),
            (1, sym.And(_t > _sq_t0, _t < _sq_t1)),
            (_gaussian_redge, sym.And(_t >= _sq_t1, _t <= _support)),
            (0, True)
        )
    )

    consts_expr = sym.And(_sigma > 0, _width >= 0, _duration >= _support, _support >= _width)
    valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
    if ncomp > 1:
        for icomp in range(ncomp):
            valid_amp_conditions_expr = sym.And(valid_amp_conditions_expr,
                                                _fractions[icomp] >= 0.,
                                                _fractions[icomp] <= 1.)

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
