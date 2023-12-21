"""Rough DRAG beta parameter calibration for X12 and SX12 gates."""

from collections.abc import Iterable, Sequence
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import RZGate
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel
from qiskit.result import Counts
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.library import RoughDrag
from qiskit_experiments.library.characterization.analysis import DragCalAnalysis

from ..constants import DEFAULT_SHOTS
from ..experiment_mixins.ef_space import EFSpaceExperiment
from ..experiment_mixins.map_to_physical_qubits import MapToPhysicalQubits
from ..gates import QutritGate, RZ12Gate
from ..util.dummy_data import ef_memory, single_qubit_counts


class EFRoughDrag(MapToPhysicalQubits, EFSpaceExperiment, RoughDrag):
    """DRAG β parameter scan for the X12 gate.

    The original RoughDrag uses the rz gate in its circuits. We replace it, in a rather hacky way,
    with rz12 to avoid having to rewrite largely identical code. Note that this can cause problems
    if the 0<->1 space rz gate is also to be used for some reason.

    Also note that sandwiching the x12 gate with Rz12(pi) is wrong in principle, as that leaves a
    geometric phase between |0> and |1>-|2> spaces. The correct sequence for a inverted-sign Rx gate
    is Rz12(-pi).Rx.Rz12(pi). We are saved by the fact that the experiment operates completely in
    the |1>-|2> space, rendering the geometric phase global.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        betas: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None,
    ):
        super().__init__(physical_qubits, schedule, betas=betas, backend=backend)
        self.analysis = DragCalAnalysisWithAbort()

    def circuits(self) -> list[QuantumCircuit]:
        """Replace the rz instructions with rz12."""
        circuits = super().circuits()

        for circuit in circuits:
            for inst in circuit.data:
                op = inst.operation
                if isinstance(op, RZGate):
                    inst.operation = RZ12Gate(op.params[0])
                elif op.name.startswith('Drag'):
                    inst.operation = QutritGate(op.name, 1, list(op.params))

        return circuits

    def dummy_data(
        self,
        transpiled_circuits: list[QuantumCircuit]
    ) -> Union[list[np.ndarray], list[Counts]]:
        """Return dummy memory or counts depending on the presence of the discriminator."""
        reps = np.asarray(self.experiment_options.reps)
        betas = np.asarray(self.experiment_options.betas)
        shots = self.run_options.get('shots', DEFAULT_SHOTS)
        one_probs = 0.4 * np.cos(2. * np.pi * 0.02 * reps[:, None] * (betas[None, :] - 2.3)) + 0.6
        #num_qubits = transpiled_circuits[0].num_qubits
        num_qubits = 1

        if self.run_options.meas_level == MeasLevel.KERNELED:
            return ef_memory(one_probs, shots, num_qubits)
        return single_qubit_counts(one_probs, shots, num_qubits)


class EFRoughDragCal(BaseCalibrationExperiment, EFRoughDrag):
    """Calibration experiment for EFRoughDrag."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        schedule_name: str = "x12",
        betas: Iterable[float] = None,
        cal_parameter_name: Optional[str] = "beta",
        auto_update: bool = True
    ):
        sched = get_qutrit_pulse_gate(schedule_name, physical_qubits[0], self._backend, self._cals,
                                      assign_params={cal_parameter_name: Parameter("beta")})

        super().__init__(
            calibrations,
            physical_qubits,
            sched,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            updater=EFRoughDragUpdater,
            auto_update=auto_update,
            betas=betas,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> dict[str, any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata['cal_param_value'] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            schedule=self._sched_name,
            group='default'
        )

        return metadata


class DragCalAnalysisWithAbort(DragCalAnalysis):
    """DragCalAnalysis with possible abort."""
    @staticmethod
    def abort_if_beta_too_large(params, iteration, resid, *args, **kwargs): # pylint: disable=unused-argument
        if abs(params['beta']) > 20. or iteration > 1000000:
            return True
        return False

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, list[curve.FitOptions]]:
        options = super()._generate_fit_guesses(user_opt, curve_data)

        if isinstance(options, curve.FitOptions):
            options.add_extra_options(iter_cb=self.abort_if_beta_too_large)
        else:
            for opt in options:
                opt.add_extra_options(iter_cb=self.abort_if_beta_too_large)

        return options


class EFRoughDragUpdater(BaseUpdater):
    __fit_parameter__ = 'beta'
