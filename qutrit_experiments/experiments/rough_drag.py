from typing import Dict, Iterable, Optional, List, Union, Sequence
import numpy as np
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.framework import Options, ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.library import RoughDrag
from qiskit_experiments.library.characterization.analysis import DragCalAnalysis
import qiskit_experiments.curve_analysis as curve

from ..common.iq_classification import IQClassification
from ..common.ef_space import EFSpaceExperiment
from ..common.transpilation import map_to_physical_qubits
from ..common.util import default_shots
from .dummy_data import ef_memory, single_qubit_counts

class EFRoughDrag(EFSpaceExperiment, IQClassification, RoughDrag):
    """DRAG β parameter scan for the X12 gate.

    The original RoughDrag uses the rz gate in its circuits. We replace it, in a rather hacky way, with
    rz12 to avoid having to rewrite largely identical code. Note that this can cause problems if the
    0<->1 space rz gate is also to be used for some reason.

    Also note that sandwiching the x12 gate with Rz12(pi) is wrong in principle, as that
    leaves a geometric phase between |0> and |1>-|2> spaces. The correct sequence for a
    inverted-sign Rx gate is Rz12(-pi).Rx.Rz12(pi). We are saved by the fact that the
    experiment operates completely in the |1>-|2> space, rendering the geometric phase
    global.
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

    def circuits(self) -> List[QuantumCircuit]:
        """Replace the rz instructions with rz12."""
        circuits = super().circuits()

        for circuit in circuits:
            for inst in circuit.data:
                if inst.operation.name == 'rz':
                    inst.operation.name = 'rz12'

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        # We can optimize this later
        return list(
            map_to_physical_qubits(circuit, self.physical_qubits, self.transpile_options.target)
            for circuit in self.circuits()
        )

    def dummy_data(
        self,
        transpiled_circuits: List[QuantumCircuit]
    ) -> Union[List[np.ndarray], List[Counts]]:
        """Return dummy memory or counts depending on the presence of the discriminator."""
        reps = np.asarray(self.experiment_options.reps)
        betas = np.asarray(self.experiment_options.betas)
        shots = self.run_options.get('shots', default_shots)
        one_probs = 0.4 * np.cos(2. * np.pi * 0.02 * reps[:, None] * (betas[None, :] - 2.3)) + 0.6
        #num_qubits = transpiled_circuits[0].num_qubits
        num_qubits = 1

        if self.run_options.meas_level == MeasLevel.KERNELED:
            return ef_memory(one_probs, shots, num_qubits)
        else:
            return single_qubit_counts(one_probs, shots, num_qubits)


class EFRoughDragCal(BaseCalibrationExperiment, EFRoughDrag):
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
        assign_params = {cal_parameter_name: Parameter("beta")}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits[0],
                                             assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            updater=EFRoughDragUpdater,
            auto_update=auto_update,
            betas=betas,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, any]:
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

    def update_calibrations(self, experiment_data: ExperimentData):
        self._updater.update(
            self._cals,
            experiment_data,
            parameter=self._param_name,
            schedule=self._sched_name,
            group=self.experiment_options.group
        )


class DragCalAnalysisWithAbort(DragCalAnalysis):
    @staticmethod
    def abort_if_beta_too_large(params, iter, resid, *args, **kwargs):
        if abs(params['beta']) > 20. or iter > 1000000:
            return True

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        options = super()._generate_fit_guesses(user_opt, curve_data)

        if isinstance(options, curve.FitOptions):
            options.add_extra_options(iter_cb=DragCalAnalysisWithAbort.abort_if_beta_too_large)
        else:
            for opt in options:
                opt.add_extra_options(iter_cb=DragCalAnalysisWithAbort.abort_if_beta_too_large)

        return options


class EFRoughDragUpdater(BaseUpdater):
    __fit_parameter__ = 'beta'
