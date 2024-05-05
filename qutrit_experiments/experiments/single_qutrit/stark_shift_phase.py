"""Measurement of AC Stark shift-induced delta parameters."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...framework.calibration_updaters import DeltaUpdater
from ...gates import X12Gate, SX12Gate, RZ12Gate
from ..phase_shift import PhaseShiftMeasurement

twopi = 2. * np.pi


class StarkCalMixin(BaseCalibrationExperiment):
    """Mixin to calibrationalize Stark shift measurements."""
    _cal_parameter_name = None
    _updater = None

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=None,
            cal_parameter_name=self._cal_parameter_name,
            updater=self._updater,
            auto_update=auto_update,
            phase_shifts=phase_shifts,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass
    

class X12QubitPhaseRotation(PhaseShiftMeasurement):
    r"""Experiment to measure the phase shift on :math:`|0\rangle` by X12.

    See the docstring of calibrations.add_x12_sx12() for the definition of :math:`\delta` and the
    :math:`\Xi` gate implementation. This phase rotation sequence results in

    .. math::

        | \psi \rangle & = P0(\delta_{\Xi}'-\pi) U_{\xi}(\pi; \delta) U_{\xi}(\pi; \delta) \sqrt{X}
                           | 0 \rangle \\
                       & ~ | 0 \rangle - i e^{i(-\delta_{\Xi}' + \delta)} | 1 \rangle,

    where :math:`\delta_{\Xi}'` is the current correction.
    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.sx(0)
        template.append(X12Gate(), [0])
        template.append(X12Gate(), [0])
        return template


class X12StarkShiftUpdater(DeltaUpdater):
    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        return (BaseUpdater.get_value(exp_data, 'phase_offset', index) + np.pi) % twopi - np.pi


class X12StarkShiftPhaseCal(StarkCalMixin, X12QubitPhaseRotation):
    """Calibration experiment for X12QubitPhaseRotation."""
    _cal_parameter_name = 'delta_x12'
    _updater = X12StarkShiftUpdater


class SX12QubitPhaseRotation(PhaseShiftMeasurement):
    r"""Experiment to measure the phase shift on :math:`|0\rangle` by SX12.

    See the docstring of calibrations.add_x12_sx12() for the definition of :math:`\delta` and the
    :math:`S\Xi` gate implementation. This phase rotation sequence results in

    .. math::

        | \psi \rangle & = P0(\delta_{S\Xi}'-\pi/2) \mathrm{diag}(1, i, -i) U_{\xi}(\pi/2; \delta)
                           \mathrm{diag}(1, i, -i) U_{\xi}(\pi/2; \delta) \sqrt{X} | 0 \rangle \\
                       & ~ | 0 \rangle - i e^{i(-\delta_{S\Xi}' + \delta - pi/2)} | 1 \rangle,

    where :math:`\delta_{S\Xi}'` is the current correction.
    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.sx(0)
        template.append(SX12Gate(), [0])
        template.append(RZ12Gate(np.pi), [0])
        template.append(SX12Gate(), [0])
        template.append(RZ12Gate(np.pi), [0])
        return template
    

class SX12StarkShiftUpdater(DeltaUpdater):
    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        delta = BaseUpdater.get_value(exp_data, 'phase_offset', index) + np.pi / 2.
        return (delta + np.pi) % twopi - np.pi


class SX12StarkShiftPhaseCal(StarkCalMixin, SX12QubitPhaseRotation):
    """Calibration experiment for SX12QubitPhaseRotation."""
    _cal_parameter_name = 'delta_sx12'
    _updater = SX12StarkShiftUpdater


class XQutritPhaseRotation(PhaseShiftMeasurement):
    r"""Experiment to measure the phase shift on :math:`|2\rangle` by X.

    The phase shift on :math:`|0\rangle` by X12 must be corrected already. See the docstring of
    calibrations.add_x12_sx12() for the definition of :math:`\delta` and the :math:`X` gate
    implementation. This phase rotation sequence results in

    .. math::

        | \psi \rangle & = P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           P2(\delta_X'/2) U_{x}(\pi; \delta)
                           P0(\delta_{S\Xi}'/2) U_{\xi}(\pi/2; \delta_{S\Xi})
                           P2(\delta_X'/2) U_{x}(\pi; \delta) | 0 \rangle \\
                       & = P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           P2(\delta_X'/2) U_{x}(\pi; \delta)
                           \left[-\frac{i}{\sqrt{2}} (| 1 \rangle - i| 2 \rangle) \right]
                       & ~ P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           \frac{1}{\sqrt{2}} (|0\rangle
                                               + e^{\frac{i}{2}(\delta_X' - \delta)}|2\rangle)
                       & = \frac{1}{\sqrt{2}} (|0\rangle
                                               - i e^{\frac{i}{2}(\delta_X' - \delta)}|1\rangle).

    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.x(0)
        template.append(X12Gate(), [0])
        return template
    

class XStarkShiftUpdater(DeltaUpdater):
    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        delta = -2. * BaseUpdater.get_value(exp_data, 'phase_offset', index)
        return (delta + np.pi) % twopi - np.pi


class XStarkShiftPhaseCal(StarkCalMixin, XQutritPhaseRotation):
    """Calibration experiment for XQutritPhaseRotation."""
    _cal_parameter_name = 'delta_x'
    _updater = XStarkShiftUpdater


class SXQutritPhaseRotation(PhaseShiftMeasurement):
    r"""Experiment to measure the phase shift on :math:`|2\rangle` by SX.

    The phase shift on :math:`|0\rangle` by X12 must be corrected already. See the docstring of
    calibrations.add_x12_sx12() for the definition of :math:`\delta` and the :math:`SX` gate
    implementation. This phase rotation sequence results in

    .. math::

        | \psi \rangle & = P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           \left[P2(\delta_{SX}'/2) U_{x}(\pi/2; \delta)\right]^2
                           P0(\delta_{S\Xi}'/2) U_{\xi}(\pi/2; \delta_{S\Xi})
                           P2(\delta_X'/2) U_{x}(\pi; \delta) | 0 \rangle \\
                       & = P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           P2(\delta_{SX}') \left[U_{x}(\pi/2; \delta)\right]^2
                           \left[-\frac{i}{\sqrt{2}} (| 1 \rangle - i| 2 \rangle) \right]
                       & ~ P0(\delta_{\Xi}/2) U_{\xi}(\pi; \delta_{\Xi})
                           \frac{1}{\sqrt{2}} (|0\rangle
                                               + e^{i(\delta_{SX}' - \delta)}|2\rangle)
                       & = \frac{1}{\sqrt{2}} (|0\rangle
                                               - i e^{i(\delta_{SX}' - \delta)}|1\rangle).

    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.sx(0)
        template.sx(0)
        template.append(X12Gate(), [0])
        return template
    

class SXStarkShiftUpdater(DeltaUpdater):
    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        delta = -1. * BaseUpdater.get_value(exp_data, 'phase_offset', index)
        return (delta + np.pi) % twopi - np.pi


class SXStarkShiftPhaseCal(StarkCalMixin, XQutritPhaseRotation):
    """Calibration experiment for XQutritPhaseRotation."""
    _cal_parameter_name = 'delta_sx'
    _updater = SXStarkShiftUpdater
