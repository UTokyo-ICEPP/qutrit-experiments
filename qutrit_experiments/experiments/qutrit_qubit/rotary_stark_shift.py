"""Stark shift correction measurement for CR rotary pulse."""
from collections.abc import Sequence
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.circuit import Gate
from qiskit.providers import Backend
from qiskit_experiments.framework import Options, ExperimentData
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...gates import X12Gate, SX12Gate
from ..single_qutrit.stark_shift_phase import BasePhaseRotation, UpdateStarkDelta

twopi = 2. * np.pi


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
        super().__init__(physical_qubits, measured_logical_qubit=1, phase_shifts=phase_shifts,
                         backend=backend)
        if schedule:
            self.set_experiment_options(schedule=schedule)

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        rotary_gate = Gate('rotary', 1, [])
        template = QuantumCircuit(2)
        template.x(1)
        template.append(SX12Gate(), [1])
        template.append(rotary_gate, [1])
        template.rz(np.pi, 1)
        template.append(rotary_gate, [1])
        template.rz(-np.pi, 1)
        template.x(1)
        template.append(X12Gate(), [1])

        template.add_calibration(rotary_gate, self.physical_qubits[1:],
                                 self.experiment_options.schedule)

        return template


class RotaryStarkShiftPhaseCal(UpdateStarkDelta, RotaryQutritPhaseRotation):
    """Calibration experiment for RotaryQutritPhaseRotation."""
    _cal_parameter_name = 'delta_rzx45p_rotary'

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
        super().__init__(
            physical_qubits,
            calibrations,
            phase_shifts=phase_shifts,
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            schedule_name=schedule_name,
            auto_update=auto_update
        )

        qubits = tuple(physical_qubits)
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
