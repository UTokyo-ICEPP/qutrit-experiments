from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from matplotlib.figure import Figure
import numpy as np
import lmfit
from uncertainties import ufloat, unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter, Delay, Gate, Measure
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.pulse.channels import PulseChannel
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.curve_analysis.utils import convert_lmfit_result, eval_with_uncertainties
from qiskit_experiments.framework import AnalysisResultData, BackendData, ExperimentData, Options
from qiskit_experiments.framework.matplotlib import default_figure_canvas
from qiskit_experiments.library import RamseyXY
from qiskit_experiments.library.characterization import RamseyXYAnalysis
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..calibrations.pulse_library import ModulatedGaussianSquare
from ..common.framework_overrides import BatchExperiment, CompoundAnalysis
from ..common.gates import SetF12, X12Gate
from ..common.util import default_shots
from .dummy_data import single_qubit_counts

twopi = 2. * np.pi


class QutritRamseyXY(RamseyXY):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delay_schedule = None
        options.reverse_qubit_order = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        delays: Optional[List] = None,
        osc_freq: Optional[float] = None,
        delay_schedule: Optional[ScheduleBlock] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        if osc_freq is None:
            osc_freq = super()._default_experiment_options().osc_freq

        super().__init__(physical_qubits, backend=backend, delays=delays, osc_freq=osc_freq)
        self.control_state = control_state
        self.set_experiment_options(delay_schedule=delay_schedule)
        if extra_metadata is None:
            self.extra_metadata = {'control_state': control_state}
        else:
            self.extra_metadata = dict(control_state=control_state, **extra_metadata)

        self.experiment_index = experiment_index

    def _pre_circuit(self) -> QuantumCircuit:
        circuit = QuantumCircuit(2)
        if self.control_state == 1:
            circuit.x(1)
        elif self.control_state == 2:
            circuit.append(SetF12(), [1])
            circuit.x(1)
            circuit.append(X12Gate(), [1])
        circuit.barrier()
        return circuit

    def _metadata(self):
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)

        return metadata

    def circuits(self) -> List[QuantumCircuit]:
        circuits = super().circuits()

        if self.experiment_options.reverse_qubit_order:
            reversed_circuits = []
            for circuit in circuits:
                reversed_circuit = circuit.copy_empty_like()
                reversed_circuit.compose(circuit, qubits=[1, 0], inplace=True)
                reversed_circuits.append(reversed_circuit)

            circuits = reversed_circuits
            control_qubit = 0
            target_qubit = 1
        else:
            control_qubit = 1
            target_qubit = 0

        for circuit in circuits:
            if self.control_state != 0:
                # Remove the measure instruction for control
                idx = next(idx for idx, inst in enumerate(circuit.data)
                           if isinstance(inst.operation, Measure) and
                              inst.qubits[0] == circuit.qregs[0][control_qubit])
                circuit.data.remove(idx)

            circuit.metadata['readout_qubits'] = [self.physical_qubits[target_qubit]]
            if self.experiment_index is not None:
                circuit.metadata['experiment_index'] = self.experiment_index

        if (delay_sched := self.experiment_options.delay_schedule):
            delay_param = delay_sched.parameters('delay')[0]

            for circuit in circuits:
                idx = next(idx for idx, inst in enumerate(circuit.data)
                           if isinstance(inst.operation, Delay))
                circuit.data[idx] = Gate(delay_sched.name, 1, [delay])
                circuit.add_calibration(delay_sched.name, [self.physical_qubits[target_qubit]],
                                        delay_sched.assign_parameters({delay_param: delay}), [delay])

        return circuits

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[np.ndarray]:
        shots = self.run_options.get('shots', default_shots)
        num_qubits = 1

        tau = 1.e-5
        freq = 2.e+5 + self.experiment_options.osc_freq
        amp = 0.49
        base = 0.51
        phase = 0.1

        delays = np.array(self.experiment_options.delays)

        p_ground_x = 1. - (amp * np.exp(-delays / tau) * np.cos(twopi * freq * delays + phase) + base)
        p_ground_y = 1. - (amp * np.exp(-delays / tau) * np.sin(twopi * freq * delays + phase) + base)
        p_ground = np.empty(2 * delays.shape[0])
        p_ground[::2] = p_ground_x
        p_ground[1::2] = p_ground_y

        return single_qubit_counts(p_ground, shots, num_qubits)


class QutritZZRamsey(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Optional[List] = None,
        osc_freq: Optional[float] = None,
        delay_schedule: Optional[ScheduleBlock] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []
        for control_state in range(3):
            exp = QutritRamseyXY(physical_qubits, control_state, delays=delays, osc_freq=osc_freq,
                                 delay_schedule=delay_schedule, backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend, analysis=QutritZZRamseyAnalysis(analyses))

        self.extra_metadata = extra_metadata

    def _metadata(self):
        metadata = super()._metadata()
        metadata.update(self.extra_metadata)

        return metadata


class QutritZZRamseyAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def __init__(
        self,
        analyses: List[RamseyXYAnalysis]
    ):
        super().__init__(analyses)

        self.figure = Figure(figsize=[9.6, 9.6])
        _ = default_figure_canvas(self.figure)
        axs = self.figure.subplots(3, 1, sharex=True)
        for control_state, (analysis, ax) in enumerate(zip(analyses, axs)):
            analysis.plotter.set_options(axis=ax)
            analysis.plotter.set_figure_options(
                figure_title=fr'Control: $|{control_state}\rangle$'
            )

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        for analysis in self._analyses:
            analysis.options.plot = self.options.plot

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List[Figure]
    ) -> Tuple[List[AnalysisResultData], List[Figure]]:
        """"""
        component_index = experiment_data.metadata["component_child_index"]

        omega_zs_by_state = []

        for control_state in range(3):
            child_data = experiment_data.child_data(component_index[control_state])
            omega_zs_by_state.append(child_data.analysis_results('freq').value * twopi)

        omega_zs_by_state = np.array(omega_zs_by_state)

        op_to_state = np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]])
        # Multiply by factor two to obtain omega_[Izζ]z
        omega_zs = (np.linalg.inv(op_to_state) @ omega_zs_by_state) * 2.

        analysis_results.append(
            AnalysisResultData(name='omega_zs', value=omega_zs)
        )

        if self.options.plot:
            figures.append(self.figure)

        return analysis_results, figures
