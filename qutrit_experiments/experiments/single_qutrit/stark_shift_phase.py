"""Measurement of AC Stark shift-induced delta parameters."""
from collections.abc import Sequence
from typing import Any, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.circuit import Parameter, Gate
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.framework import BaseExperiment, Options, ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...constants import DEFAULT_SHOTS
from ...experiment_mixins import MapToPhysicalQubitsCommonCircuit
from ...gates import X12Gate, SX12Gate, RZ12Gate
from ...util.dummy_data import single_qubit_counts

twopi = 2. * np.pi


class BasePhaseRotation(MapToPhysicalQubitsCommonCircuit, BaseExperiment):
    r"""Phase rotation-Rz-SX experiment.

    Subclasses should implement _phase_rotation_sequence that returns a circuit whose final state
    is of form

    .. math::

        |\psi\rangle = \frac{1}{\sqrt{2}} (|0\rangle - i e^{i\kappa} |1\rangle).

    The probability of outcome '1' in this experiment is then

    .. math::

        P(1) & = |\langle 1 | \sqrt{X} R_z(\phi) | \psi \rangle|^2 \\
             & = \frac{1}{2} \left( 1 + \cos(\phi + \kappa) \right).

    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.phase_shifts = np.linspace(0., twopi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=curve.OscillationAnalysis(),
                         backend=backend)

        self.analysis.set_options(
            result_parameters=[curve.ParameterRepr('phase', 'phase_offset')],
            bounds={'amp': (0., 1.)},
            fixed_parameters={'freq': 1. / (2. * np.pi)},
            normalization=False
        )
        self.analysis.plotter.set_figure_options(
            xlabel='Phase shift',
            ylabel='$P(1)$',
            ylim=(-0.1, 1.1)
        )

        if phase_shifts is not None:
            self.set_experiment_options(phase_shifts=phase_shifts)

    def circuits(self) -> list[QuantumCircuit]:
        phi = Parameter('phi')

        template = QuantumCircuit(1)
        template.compose(self._phase_rotation_sequence(), inplace=True)
        template.rz(phi, 0)
        template.sx(0)
        template.measure_all()

        template.metadata = {
            'qubits': self.physical_qubits
        }

        circuits = []
        for phase in self.experiment_options.phase_shifts:
            circuit = template.assign_parameters({phi: phase}, inplace=False)
            circuit.metadata['xval'] = phase
            circuits.append(circuit)

        return circuits

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        return QuantumCircuit(1)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]: # pylint: disable=unused-argument
        phases = self.experiment_options.phase_shifts + 0.1
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        num_qubits = 1

        one_probs = np.cos(phases) * 0.49 + 0.51

        return single_qubit_counts(one_probs, shots, num_qubits)


class UpdateStarkDelta(BaseCalibrationExperiment):
    """Mixin to calibrationize standard-gate BasePhaseRotation experiments."""
    _cal_parameter_name = 'delta'

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = None,
        schedule_name: Optional[str] = None,
        auto_update: bool = True
    ):
        if not cal_parameter_name:
            cal_parameter_name = self._cal_parameter_name

        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=schedule_name,
            phase_shifts=phase_shifts,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            schedule=self._sched_name,
            group=self.experiment_options.group
        )
        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        # Extracted phase shift + current correction
        delta = (self._extract_delta(experiment_data) + np.pi) % twopi - np.pi
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, delta, self._param_name, schedule=self._sched_name,
            group=self.experiment_options.group
        )

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        raise NotImplementedError('ABC')


class X12QubitPhaseRotation(BasePhaseRotation):
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


class X12StarkShiftPhaseCal(UpdateStarkDelta, X12QubitPhaseRotation):
    """Calibration experiment for X12QubitPhaseRotation."""
    _cal_parameter_name = 'delta_x12'

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        """See the docstring of X12QubitPhaseRotation."""
        return (BaseUpdater.get_value(experiment_data, 'phase_offset')
                + experiment_data.metadata['cal_param_value'])


class SX12QubitPhaseRotation(BasePhaseRotation):
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


class SX12StarkShiftPhaseCal(UpdateStarkDelta, SX12QubitPhaseRotation):
    """Calibration experiment for SX12QubitPhaseRotation."""
    _cal_parameter_name = 'delta_sx12'

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        """See the docstring of SX12QubitPhaseRotation."""
        return (BaseUpdater.get_value(experiment_data, 'phase_offset')
                + experiment_data.metadata['cal_param_value'] + np.pi / 2.)


class XQutritPhaseRotation(BasePhaseRotation):
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


class XStarkShiftPhaseCal(UpdateStarkDelta, XQutritPhaseRotation):
    """Calibration experiment for XQutritPhaseRotation."""
    _cal_parameter_name = 'delta_x'

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        """See the docstring of XQutritPhaseRotation."""
        return (-2. * BaseUpdater.get_value(experiment_data, 'phase_offset')
                + experiment_data.metadata['cal_param_value'])


class SXQutritPhaseRotation(BasePhaseRotation):
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


class SXStarkShiftPhaseCal(UpdateStarkDelta, SXQutritPhaseRotation):
    """Calibration experiment for SXQutritPhaseRotation."""
    _cal_parameter_name = 'delta_sx'

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        """See the docstring of XQutritPhaseRotation."""
        return (-BaseUpdater.get_value(experiment_data, 'phase_offset')
                + experiment_data.metadata['cal_param_value'])


class RotaryQutritPhaseRotation(BasePhaseRotation):
    r"""Experiment to measure the phase shift on :math:`|2\rangle` by the rotary tone of
    qubit-qutrit CR.

    The phase shift on :math:`|0\rangle` by X and X12 must be corrected already.

    For

    .. math::

        U_{x}(\theta; \delta) =
            \begin{pmatrix}
            \cos \frac{\theta}{2}    & -i \sin \frac{\theta}{2} & 0 \\
            -i \sin \frac{\theta}{2} & \cos \frac{\theta}{2} & 0 \\
            0                        & 0                     & e^{-i\delta/2} \\
            \end{pmatrix},

    the gate sequence in this experiment yields

    .. math::

        | \psi \rangle & = \Xi X P2(-\delta) S\Xi X | 0 \rangle \\
                       & ~ \frac{1}{\sqrt{2}} (|0\rangle
                                               - i e^{-i\delta}|1\rangle).

    Current correction is assumed to be always 0.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: Optional[Union[Schedule, ScheduleBlock]] = None,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, phase_shifts=phase_shifts, backend=backend)
        if schedule:
            self.set_experiment_options(schedule=schedule)

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        rotary_gate = Gate('rotary', 1, [])
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.append(rotary_gate, [0])
        template.rz(np.pi, 0)
        template.append(rotary_gate, [0])
        template.rz(-np.pi, 0)
        template.x(0)
        template.append(X12Gate(), [0])

        template.add_calibration(rotary_gate, self.physical_qubits,
                                 self.experiment_options.schedule)

        return template


class RotaryStarkShiftPhaseCal(UpdateStarkDelta, RotaryQutritPhaseRotation):
    """Calibration experiment for RotaryQutritPhaseRotation."""
    _cal_parameter_name = 'delta_rzx45p_rotary'

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        control_qubit: int,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = None,
        schedule_name: Optional[str] = None,
        auto_update: bool = True
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            phase_shifts=phase_shifts,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            schedule_name=schedule_name,
            auto_update=auto_update
        )

        qubits = (control_qubit, physical_qubits[0])
        try:
            twoq_sched = backend.target['ecr'][qubits].calibration
        except KeyError:
            twoq_sched = backend.target['cx'][qubits].calibration
        rotary = next(inst for _, inst in twoq_sched.instructions
                      if inst.name.startswith('CR90p_d'))
        with pulse.build(name='rotary') as sched:
            pulse.play(rotary.pulse, rotary.channel)

        self.set_experiment_options(schedule=sched)

    def _extract_delta(self, experiment_data: ExperimentData) -> float:
        """See the docstring of RotaryQutritPhaseRotation."""
        return -BaseUpdater.get_value(experiment_data, 'phase_offset')
