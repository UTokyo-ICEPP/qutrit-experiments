from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ...gates import X12Gate
from ..qutrit_qubit_qpt import QutritQubitQPTScan, QutritQubitQPTScanAnalysis


def make_rcr_circuit(
    physical_qubits: Sequence[int],
    cr_schedule: ScheduleBlock,
    rcr_type: str
) -> QuantumCircuit:
    params = cr_schedule.parameters

    rcr_circuit = QuantumCircuit(2)
    if rcr_type == 'x':
        rcr_circuit.x(0)
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.x(0)
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
    else:
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])
        rcr_circuit.append(Gate('cr', 2, params), [0, 1])
        rcr_circuit.append(X12Gate(), [0])

    rcr_circuit.add_calibration('cr', physical_qubits, cr_schedule, params)

    return rcr_circuit


class RepeatedCRRotaryAmplitude(QutritQubitQPTScan):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.05, 0.05, 12)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedule: ScheduleBlock,
        rcr_type: str,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits, make_rcr_circuit(physical_qubits, cr_schedule, rcr_type),
                         amp_param_name, amplitudes, angle_param_name=angle_param_name,
                         backend=backend)
        self.analysis = RepeatedCRRotaryAmplitudeAnalysis(
            [exp.analysis for exp in self._experiments]
        )


class RepeatedCRRotaryAmplitudeAnalysis(QutritQubitQPTScanAnalysis):
    """Analysis for CycledRepeatedCRWidth."""
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                    analysis_results, figures)

        amplitudes = next(res.value for res in analysis_results if res.name == 'counter_amp')
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        # Sort the amplitudes by max of |Y| over control states, then find the first local minimum
        # of max |Z|
        max_y = np.max(np.abs(unitaries[:, :, 1]), axis=1)
        max_z = np.max(np.abs(unitaries[:, :, 2]), axis=1)
        sort_by_y = np.argsort(max_y)
        max_z_sorted = max_z[sort_by_y]
        iamp = next(sort_by_y[idx] for idx in range(len(amplitudes) - 1)
                    if max_z_sorted[idx] < max_z_sorted[idx + 1])
        analysis_results.append(
            AnalysisResultData(name='rotary_amp', value=amplitudes[iamp])
        )
        return analysis_results, figures


class RepeatedCRRotaryAmplitudeCal(BaseCalibrationExperiment, RepeatedCRRotaryAmplitude):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        rcr_type: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['counter_amp', 'counter_sign_angle'],
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[float] = None,
        auto_update: bool = True
    ):
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name}
        if width is not None:
            assign_params['width'] = width
        cr_schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                                assign_params=assign_params)
        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedule,
            rcr_type,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amp_param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes
        )
        # Need to have circuits() return decomposed circuits because of how _transpiled_circuits
        # work in calibration experiments
        for qutrit_qpt in self.component_experiment():
            for qpt in qutrit_qpt.component_experiment():
                qpt.set_experiment_options(decompose_circuits=True)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        amplitude = experiment_data.analysis_results('rotary_amp', block=False).value
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        for pname, value in zip(self._param_name, [amplitude, angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )
