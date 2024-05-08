"""Measurement of phase entries of a diagonal unitary."""
from collections.abc import Sequence
import logging
from typing import Any, Optional
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import least_squares
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.providers.options import Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.curve_analysis.standard_analysis import OscillationAnalysis
from qiskit_experiments.data_processing import Probability
from qiskit_experiments.framework import AnalysisResultData, ExperimentData
from qiskit_experiments.framework.matplotlib import get_non_gui_ax

from ..framework.calibration_updaters import DeltaUpdater
from ..framework.compound_analysis import CompoundAnalysis
from ..framework_overrides.batch_experiment import BatchExperiment
from .phase_shift import PhaseShiftMeasurement

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class DiagonalCircuitPhaseShift(PhaseShiftMeasurement):
    """Measurement of the phase shift imparted by a diagonal circuit.

    In this experiment we fix the states of the spectator qubits to a specific computational basis
    and perform an SX-U-Rz-SX type sequence on the measured qubit. Assuming that U is diagonal and
    therefore acts as pure phases φ0 and φ1 to |0> and |1> of the measured qubit, respectively, the
    probability of observing |1> after the sequence is

    |<1|SX Rz(φ) U SX|0>|^2 = |1/2 [-i<0| + <1|] [exp(i (φ0-φ/2))|0> -i exp(i (φ1+φ/2))|1>]|^2
                            = cos^2(φ0 - φ1 - φ)/2
                            = 1/2 [1 + cos(φ + (φ1 - φ0))]
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.circuit = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        state: tuple[bool, ...],
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        """
        state: Initial state as a binary array from low to high with the measured qubit set to None.
        """
        measured_logical_qubit = state.index(None)
        super().__init__(
            physical_qubits,
            measured_logical_qubit=measured_logical_qubit,
            phase_shifts=phase_shifts,
            backend=backend
        )

        self.state = state
        self.set_experiment_options(circuit=circuit, measure_all=True)

        outcome_rev = list('1' if xup else '0' for xup in state)
        outcome_rev[measured_logical_qubit] = '1'

        self.analysis.set_options(outcome=''.join(reversed(outcome_rev)))

    def _phase_rotation_sequence(self) -> QuantumCircuit:
        num_qubits = len(self._physical_qubits)
        sequence = QuantumCircuit(num_qubits)
        for bit, xup in enumerate(self.state):
            if bit != self.experiment_options.measured_logical_qubit and xup:
                sequence.x(bit)
        sequence.sx(self.experiment_options.measured_logical_qubit)
        sequence.barrier()
        sequence.compose(self.experiment_options.circuit, inplace=True)
        return sequence

    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['state'] = list(self.state)
        return metadata


class PhaseTable(BatchExperiment):
    """Measure the phase of diagonals of a unitary.

    Each of the N*(2**(N-1)) phase shift measurement results corresponds to a phase difference of
    specific diagonal entries of U with some redundancy. Perform a least-squares fit to obtain the
    2**N - 1 diagonals (entry 0 is fixed to 0).
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        circuit: QuantumCircuit,
        backend: Optional[Backend] = None
    ):
        experiments = []

        num_qubits = len(physical_qubits)
        for qubit in range(num_qubits):
            for init in range(2 ** (num_qubits - 1)):
                state = [(init >> iq) % 2 == 1 for iq in range(num_qubits - 1)]
                state.insert(qubit, None)
                experiments.append(
                    DiagonalCircuitPhaseShift(physical_qubits, circuit, state, backend=backend)
                )

        super().__init__(experiments, backend=backend,
                         analysis=PhaseTableAnalysis([exp.analysis for exp in experiments]))


class PhaseTableAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def _set_subanalysis_options(self, experiment_data: ExperimentData):
        super()._set_subanalysis_options(experiment_data)
        # Restore the probability outcome string if overwritten by the parent analysis
        for analysis in self._analyses:
            if analysis.options.data_processor is None:
                continue
            try:
                prob = next(node for node in analysis.options.data_processor._nodes
                            if isinstance(node, Probability))
            except StopIteration:
                continue
            prob._outcome = analysis.options.outcome
            analysis.set_options(data_processor=analysis.options.data_processor)

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        idx_high = []
        idx_low = []
        phase_diffs = []
        exp_setups = []
        errs = []
        for child_data in experiment_data.child_data():
            logical_qubit = child_data.metadata['measured_logical_qubit']
            state = child_data.metadata['state']
            exp_setups.append((logical_qubit, state))
            num_qubits = len(state)
            idx = sum((int(state[iq]) << iq) for iq in range(num_qubits) if iq != logical_qubit)
            idx_low.append(idx)
            idx_high.append(idx + (1 << logical_qubit))
            fit_res = child_data.analysis_results('phase_offset').value
            phase_diffs.append(fit_res.n)
            errs.append(fit_res.std_dev)
        phase_diffs = np.array(phase_diffs)
        errs = np.array(errs)

        def fun(diagonals):
            phases = np.concatenate([[0], diagonals])
            test_diff = (phases[idx_high] - phases[idx_low] + np.pi) % twopi - np.pi
            return ((test_diff - phase_diffs + np.pi) % twopi - np.pi) / errs

        init = np.zeros(2 ** num_qubits - 1)
        for idx in range(1, 2 ** num_qubits):
            binary = ('{:0%db}' % num_qubits).format(idx)
            while binary.count('1') != 0:
                first_one = binary.index('1')
                free_qubit = num_qubits - 1 - first_one
                state = [b == '1' for b in binary[-1:first_one:-1]] + [None] + ([False] * first_one)
                iexp = next(iexp for iexp, setup in enumerate(exp_setups)
                            if setup == (free_qubit, state))
                init[idx - 1] += phase_diffs[iexp]
                binary = binary[:first_one] + '0' + binary[first_one + 1:]

        result = least_squares(fun, init)
        diagonals = np.concatenate([[0.], (result.x + np.pi) % twopi - np.pi])

        analysis_results.append(AnalysisResultData(name='phases', value=diagonals))

        if self.options.plot:
            num_states = 2 ** num_qubits
            ax = get_non_gui_ax()
            ax.scatter(np.arange(num_states), diagonals)
            ax.set_xlabel('State')
            ax.set_ylabel('Phase')
            ax.set_xticks(np.arange(num_states), labels=[f'{i}' for i in range(num_states)])
            ax.set_ylim(-np.pi, np.pi)
            figures.append(ax.get_figure())

            num_exps = len(phase_diffs)
            ax = get_non_gui_ax()
            ax.errorbar(np.arange(num_exps), (phase_diffs + np.pi) % twopi - np.pi, xerr=0.5,
                        yerr=errs, fmt='none', label='observed')
            ax.scatter(np.arange(num_exps),
                       (diagonals[idx_high] - diagonals[idx_low] + np.pi) % twopi - np.pi,
                       c='#ff7f0e', label='fit')
            ax.set_xticks(np.arange(num_exps),
                          labels=[f'{high}-{low}' for high, low in zip(idx_high, idx_low)])
            ax.axhline(0, c='black', ls='--', lw=0.2)
            ax.axhline(np.pi, c='black', ls='--', lw=0.2)
            ax.axhline(-np.pi, c='black', ls='--', lw=0.2)
            ax.legend()
            figures.append(ax.get_figure())

        return analysis_results, figures



class PhaseShiftUpdater(DeltaUpdater):
    @staticmethod
    def get_value(exp_data: ExperimentData, param_name: str, index: Optional[int] = -1) -> float:
        return (BaseUpdater.get_value(exp_data, 'phase_offset', index) + np.pi) % twopi - np.pi


class DiagonalPhaseCal(BaseCalibrationExperiment, DiagonalCircuitPhaseShift):
    """Generic calibration experiment of a phase of a diagonal circuit."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        circuit: QuantumCircuit,
        state: tuple[bool, ...],
        cal_parameter_name: str,
        schedule_name: Optional[str] = None,
        updater: type[BaseUpdater] = PhaseShiftUpdater,
        phase_shifts: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None,
        auto_update: bool = True
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            state,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            updater=updater,
            auto_update=auto_update,
            phase_shifts=phase_shifts,
            backend=backend
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass
