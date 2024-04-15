"""Drag with modulation(s)."""

from collections.abc import Sequence
from typing import Optional, Union
from qiskit.circuit.parameterexpression import ParameterExpression, ParameterValueType
from qiskit.pulse.library.symbolic_pulses import _PulseType, _lifted_gaussian, ScalableSymbolicPulse
from qiskit.utils import optionals as _optional
if _optional.HAS_SYMENGINE:
    import symengine as sym
else:
    import sympy as sym


class DoubleDrag(metaclass=_PulseType):
    """A pair of Drag pulses separated by parametrized delay."""

    alias = 'DoubleDrag'

    def __new__(
        cls,
        duration: Union[int, ParameterExpression],
        pulse_duration: ParameterValueType,
        amp: ParameterValueType,
        sigma: ParameterValueType,
        beta: ParameterValueType,
        angle: Optional[ParameterValueType] = None,
        interval: Optional[Union[ParameterValueType, Sequence[ParameterValueType]]] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ) -> ScalableSymbolicPulse:
        """Create new pulse instance using modulated_gaussian_square().
        Args:
            duration: Pulse length in terms of the sampling period `dt`.
            pulse_duration: Duration of individual Drag pulses.
            amp: The magnitude of the amplitude of the overall pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the function
                docstring for details.
            beta: The correction amplitude.
            angle: The angle of the complex amplitude of the pulse. Default value 0.
            interval: Zero-amplitude interval between the end of the first pulse and the beginning
                of the second. Set to maximum possible value if not given.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1. The default
                is ``True`` and the amplitude is constrained to 1.
        Returns:
            ScalableSymbolicPulse instance.
        Raises:
            PulseError: When width and risefall_sigma_ratio are both empty.
        """
        return double_drag(cls.alias, duration, pulse_duration, amp, sigma, beta, angle=angle,
                           interval=interval, name=name, limit_amplitude=limit_amplitude)


def double_drag(
    alias: str,
    duration: Union[int, ParameterExpression],
    pulse_duration: ParameterValueType,
    amp: ParameterValueType,
    sigma: ParameterValueType,
    beta: ParameterValueType,
    angle: Optional[ParameterValueType] = None,
    interval: Optional[Union[ParameterValueType, Sequence[ParameterValueType]]] = None,
    name: Optional[str] = None,
    limit_amplitude: Optional[bool] = None,
) -> ScalableSymbolicPulse:
    """Function to directly return a ScalableSymbolicPulse instance.

    Args:
        alias: Pulse instance alias.
        duration: Pulse length in terms of the sampling period `dt`.
        pulse_duration: Duration of individual Drag pulses.
        amp: The magnitude of the amplitude of the overall pulse.
        sigma: A measure of how wide or narrow the Gaussian risefall is.
        beta: The correction amplitude.
        angle: The angle of the complex amplitude of the pulse. Default value 0.
        interval: Zero-amplitude interval between the end of the first pulse and the beginning
                of the second. Set to maximum possible value if not given.
        name: Display name for this pulse envelope.
        limit_amplitude: If ``True``, then limit the amplitude of the waveform to 1. The default
            is ``True`` and the amplitude is constrained to 1.
    Returns:
        ScalableSymbolicPulse instance.
    """
    if interval is None:
        interval = (0, duration - 2 * pulse_duration)
    if not isinstance(interval, Sequence):
        interval = (0, interval)
    parameters = {'sigma': sigma, 'beta': beta, 'pulse_duration': pulse_duration,
                  'tau0': interval[0], 'tau1': interval[1]}
    # Prepare symbolic expressions
    _t, _duration, _pulse_duration, _amp, _sigma, _beta, _angle, _tau0, _tau1 = sym.symbols(
        't, duration, pulse_duration, amp, sigma, beta, angle, tau0, tau1'
    )

    _center1 = _tau0 + _pulse_duration / 2
    _gauss1 = _lifted_gaussian(_t, _center1, _tau0 + _pulse_duration + 1, _sigma)
    _deriv1 = -(_t - _center1) / (_sigma**2) * _gauss1
    _drag1 = _gauss1 + sym.I * _beta * _deriv1
    _center2 = _tau0 + _pulse_duration + _tau1 + _pulse_duration / 2
    _gauss2 = _lifted_gaussian(_t, _center2, _tau0 + _tau1 + 2 * _pulse_duration + 1, _sigma)
    _deriv2 = -(_t - _center2) / (_sigma**2) * _gauss2
    _drag2 = _gauss2 + sym.I * _beta * _deriv2

    envelope_expr = (
        _amp * sym.exp(sym.I * _angle)
        * sym.Piecewise(
            (_drag1, sym.And(_t > _tau0, _t <= _tau0 + _pulse_duration)),
            (_drag2, sym.And(_t > _tau0 + _pulse_duration + _tau1,
                             _t <= _tau0 + _tau1 + 2 * _pulse_duration)),
            (0, True)
        )
    )

    consts_expr = sym.And(_sigma > 0, _tau0 >= 0, _tau1 >= 0,
                          _tau0 + _tau1 + 2 * _pulse_duration <= _duration)
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
