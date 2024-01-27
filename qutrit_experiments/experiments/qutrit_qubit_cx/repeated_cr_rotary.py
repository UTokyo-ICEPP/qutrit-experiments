from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import X12Gate
from ..qutrit_qubit_qpt import QutritQubitQPT


class RepeatedCRTomography(QutritQubitQPT):
    """QPT of RCR for three control states."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.cr_schedule = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedule: ScheduleBlock,
        rcr_type: str,
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        rcr_circuit = QuantumCircuit(2)
        if rcr_type == 'x':
            rcr_circuit.x(0)
            rcr_circuit.append(Gate('cr', 2, []), [0, 1])
            rcr_circuit.x(0)
            rcr_circuit.append(Gate('cr', 2, []), [0, 1])
        else:
            rcr_circuit.append(Gate('cr', 2, []), [0, 1])
            rcr_circuit.append(X12Gate(), [0])
            rcr_circuit.append(Gate('cr', 2, []), [0, 1])
            rcr_circuit.append(X12Gate(), [0])

        rcr_circuit.add_calibration('cr', physical_qubits, cr_schedule)

        super().__init__(physical_qubits, rcr_circuit, backend=backend,
                         extra_metadata=extra_metadata)
        self.set_experiment_options(cr_schedule=cr_schedule)


class RepeatedCRRotaryAmplitude(BatchExperiment):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(-0.05, 0.05, 12)
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
        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        amp_param = cr_schedule.get_parameters(amp_param_name)[0]
        angle_param = cr_schedule.get_parameters(angle_param_name)[0]

        experiments = []
        for amplitude in amplitudes_exp:
            angle = 0. if amplitude > 0. else np.pi
            assign_params = {amp_param: np.abs(amplitude), angle_param: angle}
            sched = cr_schedule.assign_parameters(assign_params, inplace=False)
            experiments.append(
                RepeatedCRTomography(physical_qubits, sched, rcr_type,
                                     backend=backend, extra_metadata={'amplitude': amplitude})
            )

        analyses = [exp.analysis for exp in experiments]
        super().__init__(experiments, backend=backend,
                         analysis=RepeatedCRRotaryAmplitudeAnalysis(analyses))

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        circuit_lists = [self._experiments[0]._batch_circuits(to_transpile)]
        for exp in self._experiments[1:]:
            circuit_lists.append([c.copy() for c in circuit_lists[0]])
            cr_schedule = exp.experiment_options.cr_schedule
            for circuit in circuit_lists[-1]:
                circuit.calibrations['cr'][(self.physical_qubits, ())] = cr_schedule

        for index, circuit_list in enumerate(circuit_lists):
            for circuit in circuit_list:
                # Update metadata
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [circuit.metadata],
                    "composite_index": [index],
                }

        return sum(circuit_lists, [])


class RepeatedCRRotaryAmplitudeAnalysis(CompoundAnalysis):
    """Analysis for RepeatedCRRotaryAmplitude."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.data_processor = None # Needed to have DP propagated to QPT analysis
        options.plot = True
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list["matplotlib.figure.Figure"]
    ) -> tuple[list[AnalysisResultData], list["matplotlib.figure.Figure"]]:
        component_index = experiment_data.metadata['component_child_index']

        amplitudes = []
        unitaries = []
        fidelities = []
        for child_index in component_index:
            child_data = experiment_data.child_data(child_index)
            amplitudes.append(child_data.metadata['amplitude'])
            unitaries.append(child_data.analysis_results('unitary_parameters').value)
            fidelities.append(child_data.analysis_results('fidelities').value)

        amplitudes = np.array(amplitudes)
        unitaries = np.array(unitaries)
        fidelities = np.array(fidelities)

        analysis_results.extend([
            AnalysisResultData(name='amplitudes', value=amplitudes),
            AnalysisResultData(name='unitary_parameters', value=unitaries),
            AnalysisResultData(name='fidelities', value=fidelities)
        ])

        if self.options.plot:
            for iop, op in enumerate(['X', 'Y', 'Z']):
                plotter = CurvePlotter(MplDrawer())
                plotter.set_figure_options(
                    xlabel='Rotary amplitude',
                    ylabel=f'Unitary parameters',
                    ylim=(-2. * np.pi - 0.2, 2. * np.pi + 0.2)
                )
                for control_state in range(3):
                    plotter.set_series_data(
                        f'c{control_state}_{op}',
                        x_formatted=amplitudes,
                        y_formatted=unitaries[:, control_state, iop],
                        y_formatted_err=np.zeros_like(amplitudes)
                    )
                figures.append(plotter.figure())

            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Rotary amplitude',
                ylabel='QPT fidelities',
                ylim=(-0.1, 1.1)
            )
            for control_state in range(3):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=amplitudes,
                    y_formatted=fidelities[:, control_state],
                    y_formatted_err=np.zeros_like(amplitudes)
                )
            figures.append(plotter.figure())

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
        for rcr_exp in self.component_experiment():
            for qpt_exp in rcr_exp.component_experiment():
                qpt_exp.set_experiment_options(decompose_circuits=True)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        amplitudes = experiment_data.analysis_results('amplitudes', block=False).value
        unitaries = experiment_data.analysis_results('unitary_parameters', block=False).value
        # Sort the amplitudes by max of |Y| over control states, then find the first local minimum
        # of max |Z|
        max_y = np.max(np.abs(unitaries[:, :, 1]), axis=1)
        max_z = np.max(np.abs(unitaries[:, :, 2]), axis=1)
        sort_by_y = np.argsort(max_y)
        max_z_sorted = max_z[sort_by_y]
        iamp = next(sort_by_y[idx] for idx in range(len(amplitudes) - 1)
                    if max_z_sorted[idx] < max_z_sorted[idx + 1])
        amplitude = amplitudes[iamp]
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, amplitude, self._param_name[0], schedule=self._sched_name,
            group=self.experiment_options.group
        )
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, angle, self._param_name[1], schedule=self._sched_name,
            group=self.experiment_options.group
        )
