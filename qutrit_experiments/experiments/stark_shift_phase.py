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

from ..constants import DEFAULT_SHOTS
from ..gates import X12Gate, SX12Gate, RZ12Gate
from ..transpilation import replace_calibration_and_metadata
from ..util.dummy_data import single_qubit_counts


class BasePhaseRotation(BaseExperiment):
    """Phase rotation-Rz-SX experiment."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.phase_shifts = np.linspace(0., 2. * np.pi, 16, endpoint=False)
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

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        return replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                self._backend_data.coupling_map)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata

    def dummy_data(self, transpiled_circuits: list[QuantumCircuit]) -> list[Counts]:
        phases = self.experiment_options.phase_shifts + 0.1
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        num_qubits = 1

        one_probs = np.cos(phases) * 0.49 + 0.51

        return single_qubit_counts(one_probs, shots, num_qubits)


class X12QubitPhaseRotation(BasePhaseRotation):
    r"""Experiment to measure the phase shift on :math:`|0\rangle` by X12.

    For

    .. math::

        U_{\xi}(\theta; \delta) =
            \begin{pmatrix}
            e^{-i\delta/2} & 0                        & 0 \\
            0              & \cos \frac{\theta}{2}    & -i \sin \frac{\theta}{2} \\
            0              & -i \sin \frac{\theta}{2} & \cos \frac{\theta}{2}
            \end{pmatrix},

    the gate sequence in this experiment yields

    .. math::

        |\langle 1 | \sqrt{X} R_z(\phi) U_{\xi}(-\pi; \delta) U_{\xi}(\pi; \delta) \sqrt{X}
            | 0 \rangle|^2
        \propto 1 + \cos (\delta + \phi).
    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.sx(0)
        template.append(X12Gate(), [0])
        template.append(RZ12Gate(np.pi), [0])
        template.append(X12Gate(), [0])
        template.append(RZ12Gate(-np.pi), [0])

        return template


class X12StarkShiftPhaseCal(BaseCalibrationExperiment, X12QubitPhaseRotation):
    """Calibration experiment for X12QubitPhaseRotation."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = 'x12stark',
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=None,
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
            self._sched_name,
            self.experiment_options.group
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        value = BaseUpdater.get_value(experiment_data, 'phase_offset')

        # value = delta - current_correction
        delta = value + experiment_data.metadata['cal_param_value']

        while delta > np.pi:
            delta -= 2. * np.pi
        while delta <= -np.pi:
            delta += 2. * np.pi

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, delta, self._param_name,
            group=self.experiment_options.group
        )


class XQutritPhaseRotation(BasePhaseRotation):
    r"""Experiment to measure the phase shift on :math:`|2\rangle` by X.

    The phase shift on :math:`|0\rangle` by X12 must be corrected already.

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

        | \langle 1 | \sqrt{X} R_{z}(\phi) R_{\xi}(\pi) U_{x}(\pi; \delta) R_{\xi}(\pi/2) X
            | 0 \rangle |^2
        \propto 1 + \cos \left( \phi - \frac{\delta}{2} \right).
    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.x(0)
        template.append(X12Gate(), [0])

        return template


class XStarkShiftPhaseCal(BaseCalibrationExperiment, XQutritPhaseRotation):
    """Calibration experiment for XQutritPhaseRotation."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = 'xstark',
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=None,
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
            self._sched_name,
            self.experiment_options.group
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        value = BaseUpdater.get_value(experiment_data, 'phase_offset')

        # value = -delta / 2
        # Current correction value does not enter here because we use the backend default x
        delta = -2. * value

        while delta > np.pi:
            delta -= 2. * np.pi
        while delta <= -np.pi:
            delta += 2. * np.pi

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, delta, self._param_name,
            group=self.experiment_options.group
        )


class SXQutritPhaseRotation(BasePhaseRotation):
    r"""Experiment to measure the phase shift on :math:`|2\rangle` by SX.

    The phase shift on :math:`|0\rangle` by X12 must be corrected already.

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

        | \langle 1 | \sqrt{X} R_{z}(\phi) \Xi \left[U_{x}(\pi/2; \delta)\right]^2
                      \sqrt{\Xi} X | 0 \rangle |^2
        \propto 1 + \cos \left( \phi - \delta \right).
    """
    def _phase_rotation_sequence(self) -> QuantumCircuit:
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.sx(0)
        template.sx(0)
        template.append(X12Gate(), [0])

        return template


class SXStarkShiftPhaseCal(BaseCalibrationExperiment, SXQutritPhaseRotation):
    """Calibration experiment for SXQutritPhaseRotation."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'sxstark',
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=None,
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
            self._sched_name,
            self.experiment_options.group
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        value = BaseUpdater.get_value(experiment_data, 'phase_offset')

        # value = -delta
        # Current correction value does not enter here because we use the backend default x
        delta = -value

        while delta > np.pi:
            delta -= 2. * np.pi
        while delta <= -np.pi:
            delta += 2. * np.pi

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, delta, self._param_name,
            group=self.experiment_options.group
        )


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

        | \langle 1 | \sqrt{X} R_{z}(\phi) \Xi X U_{x}(-\theta; \delta) U_{x}(\theta; \delta)
            \sqrt{\Xi} X | 0 \rangle |^2
        \propto 1 + \cos \left( \phi - \delta \right).
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: Union[Schedule, ScheduleBlock],
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, phase_shifts=phase_shifts, backend=backend)

        channel = schedule.channels[0]
        with pulse.build(name='pump') as sched:
            pulse.call(schedule)
            with pulse.phase_offset(np.pi, channel):
                pulse.call(schedule)

        self.pump_sched = sched

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        gate = Gate('rotary', 1, [])
        template = QuantumCircuit(1)
        template.x(0)
        template.append(SX12Gate(), [0])
        template.append(gate, [0])
        template.x(0)
        template.append(X12Gate(), [0])

        template.add_calibration(gate, self.physical_qubits, self.pump_sched)

        return template


class RotaryStarkShiftPhaseCal(BaseCalibrationExperiment, RotaryQutritPhaseRotation):
    """Calibration experiment for RotaryQutritPhaseRotation."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        phase_shifts: Optional[Sequence[float]] = None,
        schedule_name: str = 'rzx45m_rotary',
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = 'delta',
        auto_update: bool = True
    ):
        assign_params = {cal_parameter_name: 0.}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
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
            self._sched_name,
            self.experiment_options.group
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        value = BaseUpdater.get_value(experiment_data, 'phase_offset')

        # value = -delta
        # Current correction value does not enter here because we use the backend default x
        delta = -value

        while delta > np.pi:
            delta -= 2. * np.pi
        while delta <= -np.pi:
            delta += 2. * np.pi

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, delta, self._param_name,
            schedule=self._sched_name,
            group=self.experiment_options.group
        )
