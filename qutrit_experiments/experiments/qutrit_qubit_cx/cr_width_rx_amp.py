from collections.abc import Sequence
from typing import Any, Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import XGate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.framework import (AnalysisResultData, BackendData, BackendTiming,
                                          ExperimentData, Options)
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import X12Gate
from ..qutrit_qubit_qpt import QutritQubitQPT

twopi = 2. * np.pi


class CycledRepeatedCRTomography(QutritQubitQPT):
    """Three-level QPT of CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.cr_schedules = None
        options.rx_schedule = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: str,
        backend: Optional[Backend] = None,
        extra_metadata: Optional[dict[str, Any]] = None
    ):
        crcr_circuit = QuantumCircuit(2)
        if rcr_type == 'x':
            # [X+]-[RCR-]
            crcr_circuit.append(Gate('offset_rx', 1, []), [1])
            crcr_circuit.append(X12Gate())
            crcr_circuit.append(Gate('crm', 2, []), [0, 1])
            crcr_circuit.x(0)
            crcr_circuit.append(Gate('crm', 2, []), [0, 1])
            # [X+]-[RCR+] x 2
            for _ in range(2):
                crcr_circuit.append(X12Gate())
                crcr_circuit.append(Gate('crp', 2, []), [0, 1])
                crcr_circuit.x(0)
                crcr_circuit.append(Gate('crp', 2, []), [0, 1])
        else:
            # [RCR+]-[X+] x 2
            for _ in range(2):
                crcr_circuit.append(Gate('crp', 2, []), [0, 1])
                crcr_circuit.append(X12Gate(), [0])
                crcr_circuit.append(Gate('crp', 2, []), [0, 1])
                crcr_circuit.x(0)
            # [RCR-]-[X+]
            crcr_circuit.append(Gate('crm', 2, []), [0, 1])
            crcr_circuit.append(X12Gate(), [0])
            crcr_circuit.append(Gate('crm', 2, []), [0, 1])
            crcr_circuit.x(0)
            crcr_circuit.append(Gate('offset_rx', 1, []), [1])

        crcr_circuit.add_calibration('crp', physical_qubits, cr_schedules[0])
        crcr_circuit.add_calibration('crm', physical_qubits, cr_schedules[1])
        crcr_circuit.add_calibration('offset_rx', (physical_qubits[1],), rx_schedule)

        super().__init__(physical_qubits, crcr_circuit, backend=backend,
                         extra_metadata=extra_metadata)
        self.set_experiment_options(cr_schedules=cr_schedules, rx_schedule=rx_schedule)


class CRWidthAndRxAmpScan(BatchExperiment):
    """Experiment to simultaneously scan the Rx amplitude and CR width."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.widths = np.linspace(0., 512, 6)
        options.amplitudes = np.linspace(0., 0.05, 6)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_schedule: ScheduleBlock,
        rcr_type: str,
        width_param_name: str = 'width',
        margin_param_name: str = 'margin',
        amp_param_name: str = 'amp',
        angle_param_name: str = 'sign_angle',
        widths: Optional[Sequence[float]] = None,
        amplitudes: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if (widths_exp := widths) is None:
            widths_exp = self._default_experiment_options().widths
        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        assert len(widths_exp) == len(amplitudes_exp)

        width_params = [sched.get_parameters(width_param_name)[0] for sched in cr_schedules]
        margin_params = [sched.get_parameters(margin_param_name)[0] for sched in cr_schedules]
        amp_param = rx_schedule.get_parameters(amp_param_name)[0]
        angle_param = rx_schedule.get_parameters(angle_param_name)[0]

        test_assign = {width_params[0]: 0., margin_params[0]: 0.}
        risefall_duration = cr_schedules[0].assign_parameters(test_assign, inplace=False).duration

        experiments = []
        for width, amplitude in zip(widths_exp, amplitudes_exp):
            duration = BackendTiming(backend).round_pulse(samples=(width + risefall_duration))
            margin = duration - (width + risefall_duration)
            if margin < 0.:
                margin += BackendData(backend).granularity
            cr_scheds = []
            for sched, width_param, margin_param in zip(cr_schedules, width_params, margin_params):
                assign_params = {width_param: width, margin_param: margin}
                cr_scheds.append(sched.assign_parameters(assign_params, inplace=False))

            if amplitude < 0.:
                assign_params = {amp_param: -amplitude, angle_param: np.pi}
            else:
                assign_params = {amp_param: amplitude, angle_param: 0.}
            rx_sched = rx_schedule.assign_parameters(assign_params, inplace=False)

            experiments.append(
                CycledRepeatedCRTomography(physical_qubits, cr_scheds, rx_sched, rcr_type,
                                           backend=backend,
                                           extra_metadata={'width': width, 'amplitude': amplitude})
            )

        analyses = [exp.analysis for exp in experiments]
        super().__init__(experiments, backend=backend,
                         analysis=CRWidthAndRxAmpScanAnalysis(analyses))

    def _batch_circuits(self, to_transpile=False) -> list[QuantumCircuit]:
        circuit_lists = [self._experiments[0]._batch_circuits(to_transpile)]
        for exp in self._experiments[1:]:
            circuit_lists.append([c.copy() for c in circuit_lists[0]])
            cr_schedules = exp.experiment_options.cr_schedules
            rx_schedule = exp.experiment_options.rx_schedule
            for circuit in circuit_lists[-1]:
                circuit.calibrations['crp'][(self.physical_qubits, ())] = cr_schedules[0]
                circuit.calibrations['crm'][(self.physical_qubits, ())] = cr_schedules[1]
                circuit.calibrations['offset_rx'][((self.physical_qubits[1],), ())] = rx_schedule

        for index, circuit_list in enumerate(circuit_lists):
            for circuit in circuit_list:
                # Update metadata
                circuit.metadata = {
                    "experiment_type": self._type,
                    "composite_metadata": [circuit.metadata],
                    "composite_index": [index],
                }

        return sum(circuit_lists, [])


class CRWidthAndRxAmpScanAnalysis(CompoundAnalysis):
    """Analysis for CRWidthAndRxAmpScan."""
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


class CRWidthAndRxAmpScanCal(BaseCalibrationExperiment, CRWidthAndRxAmpScan):
    """Calibration experiment for CR width and Rx amplitude"""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        rcr_type: str,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['width', 'margin', 'amp', 'sign_angle'],
        schedule_name: str = ['cr', 'cr', 'offset_rx', 'offset_rx'],
        widths: Optional[Sequence[float]] = None,
        amplitudes: Optional[Sequence[float]] = None,
        auto_update: bool = True
    ):
        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name[:2]}
        cr_schedules = [calibrations.get_schedule(schedule_name[0], physical_qubits,
                                                  assign_params=assign_params)]
        for pname in ['cr_sign_angle', 'counter_sign_angle', 'cr_stark_sign_phase']:
            # Stark phase is relative to the CR angle, and we want to keep it the same for CRp and CRm
            assign_params[pname] = np.pi
        cr_schedules.append(calibrations.get_schedule(schedule_name[0], physical_qubits,
                                                      assign_params=assign_params))

        assign_params = {pname: Parameter(pname) for pname in cal_parameter_name[2:]}
        rx_schedule = calibrations.get_schedule(schedule_name[2], physical_qubits[1],
                                                assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            rx_schedule,
            rcr_type,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            width_param_name=cal_parameter_name[0],
            margin_param_name=cal_parameter_name[1],
            amp_param_name=cal_parameter_name[2],
            widths=widths,
            amplitudes=amplitudes
        )
        # Need to have circuits() return decomposed circuits because of how _transpiled_circuits
        # work in calibration experiments
        for crcr_exp in self.component_experiment():
            for qpt_exp in crcr_exp.component_experiment():
                qpt_exp.set_experiment_options(decompose_circuits=True)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        pass