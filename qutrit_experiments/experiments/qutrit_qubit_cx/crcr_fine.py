"""Error amplification experiment to fine-calibrate the CR width and Rx amplitude."""
from collections.abc import Sequence
from numbers import Number
from typing import Any, Optional
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers.options import Options
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.curve_analysis.standard_analysis import OscillationAnalysis
from qiskit_experiments.framework import AnalysisResultData, BaseExperiment, ExperimentData
from qiskit_experiments.library.characterization.analysis import FineAmplitudeAnalysis
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...experiment_mixins import MapToPhysicalQubits
from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import X12Gate
from .util import RCRType, get_cr_schedules, get_margin, make_crcr_circuit

twopi = 2. * np.pi


class CycledRepeatedCRPingPong(MapToPhysicalQubits, BaseExperiment):
    """Error amplification experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.repetitions = np.arange(1, 9)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_angle: float,
        rcr_type: RCRType,
        cx_sign: Optional[Number] = None,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
        self._control_state = control_state
        self._crcr_circuit = make_crcr_circuit(physical_qubits, cr_schedules, rx_angle, rcr_type)
        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)

        # Fit function is P(1) = A/2*cos(n*(apg + dθ) - po) + B
        # Let Rx(θ) correspond to cos(θ - π)
        # Phase offset is π/2 with the initial SX. The sign of apg for control=1 depends on cx_sign
        angle_per_gate = np.pi
        if control_state == 1:
            angle_per_gate *= cx_sign

        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": angle_per_gate,
                "phase_offset": np.pi / 2.
            },
            outcome='1'
        )

    def circuits(self) -> list[QuantumCircuit]:
        circuits = []

        for add_x in [0, 1]:
            circuit = QuantumCircuit(2, 1)
            if self._control_state != 0:
                circuit.x(0)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0])
            if add_x == 1:
                circuit.x(1)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0]) # restore qubit space
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': add_x,
                'series': 'spam-cal'
            }
            circuits.append(circuit)

        for repetition in self.experiment_options.repetitions:
            circuit = QuantumCircuit(2, 1)
            if self._control_state != 0:
                circuit.x(0)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0])
            circuit.sx(1)
            for _ in range(repetition):
                circuit.compose(self._crcr_circuit, inplace=True)
                if self._control_state != 1:
                    circuit.x(1)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0]) # restore qubit space
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'xval': repetition,
                'series': 1
            }
            circuits.append(circuit)

        return circuits


class CycledRepeatedCRFine(BatchExperiment):
    """Perform CycledRepeatedCRPingPong for control state 0 and 1 to calibrate width and rx amp."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rx_angle: float,
        rcr_type: RCRType,
        cx_sign: Number,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = [
            CycledRepeatedCRPingPong(physical_qubits, ic, cr_schedules, rx_angle, rcr_type,
                                     cx_sign=cx_sign, repetitions=repetitions, backend=backend)
            for ic in range(2)
        ]
        super().__init__(experiments, backend=backend)


class CycledRepeatedCRFineCal(BaseCalibrationExperiment, CycledRepeatedCRFine):
    """Calibration experiment for CycledRepeatedCRFine."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.calibration_qubit_index = {}
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        dxda: np.ndarray,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['cr_amp', 'qutrit_qubit_cx_offsetrx'],
        schedule_name: list[str] = ['cr', None],
        current_cal_groups: tuple[str, str] = ('default', 'default'),
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        param_cal_groups = {cal_parameter_name[0]: current_cal_groups[0]}
        cr_schedules = get_cr_schedules(calibrations, physical_qubits,
                                        param_cal_groups=param_cal_groups)
        rx_angle = calibrations.get_parameter_value(cal_parameter_name[1], physical_qubits[1],
                                                    group=current_cal_groups[1])

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            rx_angle,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            repetitions=repetitions
        )
        self.dxda = dxda
        self.current_cal_groups = current_cal_groups

        self.set_experiment_options(calibration_qubit_index={(self._param_name[1], None): [1]})

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        d_thetas = np.array([
            experiment_data.child_data(idx).analysis_results('d_theta').value.n
            for idx in component_index
        ])
        # dθ_i = dxda_i da + dx
        # -> (da, dx)^T = ([dxda_0 1], [dxda_1 1])^{-1} (dθ_0, dθ_1)^T
        mat = np.stack([self.dxda, np.ones(2)], axis=1)
        d_amp, d_angle = np.linalg.inv(mat) @ d_thetas

        # Calculate the new CR amplitude
        current_amp = self._cals.get_parameter_value(self._param_name[0], self.physical_qubits,
                                                     schedule=self._sched_name[0],
                                                     group=self.current_cal_groups[0])
        new_amp = current_amp - d_amp
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_amp, self._param_name[0], schedule=self._sched_name[0],
            group=self.experiment_options.group
        )

        # Calculate the new Rx angle
        current_angle = self._cals.get_parameter_value(self._param_name[1], self.physical_qubits[1],
                                                       group=self.current_cal_groups[1])
        new_angle = current_angle - d_angle
        param_value = ParameterValue(
            value=new_angle,
            date_time=BaseUpdater._time_stamp(experiment_data),
            group=self.experiment_options.group,
            exp_id=experiment_data.experiment_id,
        )
        self._cals.add_parameter_value(param_value, self._param_name[1], self.physical_qubits[1])


class CycledRepeatedCRFineRxAngleCal(BaseCalibrationExperiment, CycledRepeatedCRPingPong):
    """Calibration experiment for Rx angle only."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.calibration_qubit_index = {}
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'qutrit_qubit_cx_offsetrx',
        schedule_name: Optional[str] = None,
        current_cal_group: str = 'default',
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        cr_schedules = get_cr_schedules(calibrations, physical_qubits)
        rx_angle = calibrations.get_parameter_value(cal_parameter_name, physical_qubits[1],
                                                    group=current_cal_group)

        super().__init__(
            calibrations,
            physical_qubits,
            1,
            cr_schedules,
            rx_angle,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            repetitions=repetitions
        )
        self.current_cal_group = current_cal_group
        self.set_experiment_options(calibration_qubit_index={(self._param_name, None): [1]})

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        d_theta = experiment_data.analysis_results('d_theta', block=False).value.n
        current_angle = self._cals.get_parameter_value(self._param_name, self.physical_qubits[1],
                                                       group=self.current_cal_group)
        param_value = ParameterValue(
            value=current_angle - d_theta,
            date_time=BaseUpdater._time_stamp(experiment_data),
            group=self.experiment_options.group,
            exp_id=experiment_data.experiment_id,
        )
        self._cals.add_parameter_value(param_value, self._param_name, self.physical_qubits[1])


class CycledRepeatedCRRxScan(MapToPhysicalQubits, BaseExperiment):
    """Error amplification experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: RCRType,
        angles: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=OscillationAnalysis(), backend=backend)

        self._control_state = control_state
        rx_angle = Parameter('rx_angle')
        self._crcr_circuit = make_crcr_circuit(physical_qubits, cr_schedules, rx_angle, rcr_type)
        if angles is not None:
            self.set_experiment_options(angles=angles)

        self.analysis.set_options(
            fixed_parameters={'freq': 1. / twopi},
            result_parameters=['phase'],
            outcome='1'
        )

        self.extra_metadata = {}

    def circuits(self) -> list[QuantumCircuit]:
        rx_angle = self._crcr_circuit.get_parameter('rx_angle')
        circuits = []
        for angle in self.experiment_options.angles:
            crcr = self._crcr_circuit.assign_parameters({rx_angle: angle}, inplace=False)
            circuit = QuantumCircuit(2, 1)
            if self._control_state != 0:
                circuit.x(0)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0])
            circuit.compose(crcr, inplace=True)
            if self._control_state == 2:
                circuit.append(X12Gate(), [0]) # restore qubit space
            circuit.measure(1, 0)
            circuit.metadata = {
                'qubits': self._physical_qubits,
                'control_state': self._control_state,
                'xval': angle
            }
            circuits.append(circuit)

        return circuits
    
    def _metadata(self) -> dict[str, Any]:
        metadata = super()._metadata()
        metadata['control_state'] = self._control_state
        metadata.update(self.extra_metadata)
        return metadata
    

class CycledRepeatedCRRxOffsetAmpScan(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: RCRType,
        cr_amps: Sequence[float],
        amp_param_name: str = 'cr_amp',
        angles: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        amp_params = [s.get_parameters(amp_param_name)[0] for s in cr_schedules]
        experiments = []
        for amp_val in cr_amps:
            scheds = [s.assign_parameters({p: amp_val}) for s, p in zip(cr_schedules, amp_params)]
            for control_state in range(2):
                exp = CycledRepeatedCRRxScan(physical_qubits, control_state, scheds, rcr_type,
                                             angles=angles, backend=backend)
                exp.extra_metadata['cr_amp'] = amp_val
                experiments.append(exp)

        analyses = [exp.analysis for exp in experiments]
        super().__init__(experiments, backend=backend,
                         analysis=CycledRepeatedCRRxOffsetAmpScanAnalysis(analyses))
        

class CycledRepeatedCRRxOffsetAmpScanAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        component_index = experiment_data.metadata['component_child_index']
        xvals = []
        yvals = []
        for ichild, child_index in enumerate(component_index):
            child_data = experiment_data.child_data(child_index)
            if ichild % 2 == 0:
                xvals.append(child_data.metadata['cr_amp'])
                yvals.append([child_data.analysis_results('phase').value])
            else:
                yvals[-1].append(child_data.analysis_results('phase').value)

        xvals = np.array(xvals)
        yvals = np.array(yvals)
        yvals_n = unp.nominal_values(yvals)
        yvals_e = unp.std_devs(yvals)

        def curve(x, slope, intercept):
            return slope * x + intercept

        popt_ufloats = []
        for control_state in range(2):
            p0_slope = np.diff(yvals_n[[0, -1], control_state])[0] / np.diff(xvals[[0, -1]])[0]
            p0_intercept = np.mean(yvals_n[:, control_state] - p0_slope * xvals)
            p0 = (p0_slope, p0_intercept)
            popt, pcov = curve_fit(curve, xvals, yvals_n[:, control_state],
                                   sigma=yvals_e[:, control_state], p0=p0)
            popt_ufloats.append(correlated_values(popt, pcov))
        popt_ufloats = np.array(popt_ufloats)

        analysis_results.append(AnalysisResultData(name='linear_fit_params', value=popt_ufloats))
        
        diff_slope = np.diff(popt_ufloats[:, 0])[0]
        diff_intercept = np.diff(popt_ufloats[:, 1])[0]
        amp_opt = ((np.pi - diff_intercept) / diff_slope) % (twopi / np.abs(diff_slope.n))
        # Optimal angle is where cosine at block 0 is 1
        angle_opt = -curve(amp_opt, *popt_ufloats[0])
        analysis_results.extend([
            AnalysisResultData(name='cr_amp', value=amp_opt),
            AnalysisResultData(name='qutrit_qubit_cx_offsetrx', value=angle_opt)
        ])

        if self.options.plot:
            x_interp = np.linspace(xvals[0], xvals[-1], 100)
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='CR amplitude',
                ylabel='CRCR Rx scan offsets'
            )
            for control_state in range(2):
                plotter.set_series_data(
                    f'c{control_state}',
                    x_formatted=xvals,
                    y_formatted=yvals_n[:, control_state],
                    y_formatted_err=yvals_e[:, control_state],
                    x_interp=x_interp,
                    y_interp=curve(x_interp, *unp.nominal_values(popt_ufloats[control_state]))
                )
            figures.append(plotter.figure())


class CycledRepeatedCRFineScanCal(BaseCalibrationExperiment, CycledRepeatedCRRxOffsetAmpScan):
    """Calibration experiment with CycledRepeatedCRRxOffsetAmpScan."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.calibration_qubit_index = {}
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['cr_amp', 'qutrit_qubit_cx_offsetrx'],
        schedule_name: Optional[list[str]] = ['cr', None],
        current_cal_group: str = 'default',
        angles: Optional[Sequence[float]] = None,
        auto_update: bool = True
    ):
        assign_params = {cal_parameter_name[0]: Parameter('cr_amp')}
        cr_schedules = get_cr_schedules(calibrations, physical_qubits, assign_params=assign_params)
        current_cr_amp = calibrations.get_parameter_value(cal_parameter_name[0], physical_qubits,
                                                          schedule_name[0], group=current_cal_group)
        cr_amps = np.linspace(current_cr_amp - 0.05, current_cr_amp + 0.05, 4)
        
        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            cr_amps,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            angles=angles
        )
        self.set_experiment_options(calibration_qubit_index={(self._param_name[1], None): [1]})

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        qubit_lists = [self.physical_qubits, [self.physical_qubits[1]]]
        for pname, sname, qubits in zip(self._parma_name, self._sched_name, qubit_lists):
            value = BaseUpdater.get_value(experiment_data, pname)
            param_value = ParameterValue(
                value=value,
                date_time=BaseUpdater._time_stamp(experiment_data),
                group=self.experiment_options.group,
                exp_id=experiment_data.experiment_id,
            )
            self._cals.add_parameter_value(param_value, pname, qubits, schedule=sname)
