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

from ...calibrations import get_qutrit_qubit_composite_gate
from ...experiment_mixins import MapToPhysicalQubits
from ...framework.compound_analysis import CompoundAnalysis
from ...framework_overrides.batch_experiment import BatchExperiment
from ...gates import QutritQubitCXGate, X12Gate

twopi = 2. * np.pi


class CycledRepeatedCRPingPong(MapToPhysicalQubits, BaseExperiment):
    """Error amplification experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.repetitions = np.arange(1, 9)
        options.schedule = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        rcr_type: int,
        cx_sign: Optional[Number] = None,
        schedule: Optional[ScheduleBlock] = None,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=FineAmplitudeAnalysis(), backend=backend)
        self.set_experiment_options(schedule=schedule)
        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)

        self._control_state = control_state
        self._cx_gate = QutritQubitCXGate.of_type(rcr_type)()

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
                circuit.append(self._cx_gate, [0, 1])
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
            if (sched := self.experiment_options.schedule) is not None:
                circuit.add_calibration(self._cx_gate, self.physical_qubits, sched)
            circuits.append(circuit)

        return circuits


class CycledRepeatedCRFine(BatchExperiment):
    """Perform CycledRepeatedCRPingPong for control state 0 and 1 to calibrate width and rx amp."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        rcr_type: int,
        cx_sign: Number,
        schedule: Optional[ScheduleBlock] = None,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = [
            CycledRepeatedCRPingPong(physical_qubits, ic, rcr_type, cx_sign=cx_sign,
                                     schedule=schedule, repetitions=repetitions, backend=backend)
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
        cal_parameter_name: list[str] = ['cr_amp', 'angle'],
        schedule_name: list[str] = ['cr', 'cx_offset_rx'],
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        physical_qubits = tuple(physical_qubits)
        rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
        super().__init__(
            calibrations,
            physical_qubits,
            rcr_type,
            calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            repetitions=repetitions
        )
        self.set_experiment_options(
            calibration_qubit_index={(self._param_name[1], self._sched_name[1]): [1]}
        )

        self._dxda = dxda
        self._gate_name = QutritQubitCXGate.of_type(rcr_type).gate_name
        self._schedule = get_qutrit_qubit_composite_gate(self._gate_name, physical_qubits,
                                                         calibrations, target=backend.target)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        if circuit.metadata['composite_metadata'][0]['series'] == 1:
            circuit.add_calibration(self._gate_name, self.physical_qubits, self._schedule)

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        d_thetas = np.array([
            experiment_data.child_data(idx).analysis_results('d_theta').value.n
            for idx in component_index
        ])
        # dθ_i = dxda_i da + dx
        # -> (da, dx)^T = ([dxda_0 1], [dxda_1 1])^{-1} (dθ_0, dθ_1)^T
        mat = np.stack([self._dxda, np.ones(2)], axis=1)
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
        self._cals.add_parameter_value(param_value, self._param_name[1], self.physical_qubits[1],
                                       schedule=self._sched_name[1])


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
        cal_parameter_name: str = 'angle',
        schedule_name: str = 'cx_offset_rx',
        current_cal_group: str = 'default',
        repetitions: Optional[Sequence[int]] = None,
        auto_update: bool = True
    ):
        rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
        super().__init__(
            calibrations,
            physical_qubits,
            1,
            rcr_type,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            cx_sign=calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits),
            repetitions=repetitions
        )
        self.current_cal_group = current_cal_group
        self.set_experiment_options(
            calibration_qubit_index={(self._param_name, self._sched_name): [1]}
        )

        self._gate_name = QutritQubitCXGate.of_type(rcr_type).gate_name
        self._schedule = get_qutrit_qubit_composite_gate(self._gate_name, physical_qubits,
                                                         calibrations, target=backend.target)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        if circuit.metadata['series'] == 1:
            circuit.add_calibration(self._gate_name, self.physical_qubits, self._schedule)

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
        self._cals.add_parameter_value(param_value, self._param_name, self.physical_qubits[1],
                                       schedule=self._sched_name[1])


class CycledRepeatedCRRxScan(MapToPhysicalQubits, BaseExperiment):
    """Angle-scan experiment to fine-calibrate CRCR."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.angles = np.linspace(-np.pi, np.pi, 16, endpoint=False)
        options.schedule = None
        options.gate_params = None
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int,
        rcr_type: int,
        schedule: Optional[ScheduleBlock] = None,
        angles: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, analysis=OscillationAnalysis(), backend=backend)
        self.set_experiment_options(schedule=schedule)
        if angles is not None:
            self.set_experiment_options(angles=angles)

        self._control_state = control_state
        if schedule is not None:
            self._angle_param = next(iter(schedule.parameters))
        else:
            self._angle_param = Parameter('angle')
        self._cx_gate = QutritQubitCXGate.of_type(rcr_type)(params=[self._angle_param])

        self.analysis.set_options(
            fixed_parameters={'freq': 1. / twopi},
            result_parameters=['phase'],
            outcome='1', # Measure P(1) with negative cosine amplitude
            bounds={'amp': (-1., 0.)},
            p0={'amp': -0.5}
        )

        self.extra_metadata = {}

    def circuits(self) -> list[QuantumCircuit]:
        template = QuantumCircuit(2, 1)
        if self._control_state != 0:
            template.x(0)
        if self._control_state == 2:
            template.append(X12Gate(), [0])
        template.append(self._cx_gate, [0, 1])
        if self._control_state == 2:
            template.append(X12Gate(), [0]) # restore qubit space
        template.measure(1, 0)
        template.metadata = {
            'qubits': self._physical_qubits,
            'control_state': self._control_state
        }
        if (sched := self.experiment_options.schedule) is not None:
            template.add_calibration(self._cx_gate, self.physical_qubits, sched, [self._angle_param])

        circuits = []
        for angle in self.experiment_options.angles:
            circuit = template.assign_parameters({self._angle_param: angle}, inplace=False)
            circuit.metadata['xval'] = angle
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
        rcr_type: int,
        cr_amps: Sequence[float],
        schedule: Optional[ScheduleBlock] = None,
        amp_param_name: Optional[str] = None,
        angles: Optional[Sequence[float]] = None,
        backend: Optional[Backend] = None
    ):
        if (sched_template := schedule) is not None:
            amp_param = sched_template.get_parameters(amp_param_name)[0]

        experiments = []
        for amp_val in cr_amps:
            if sched_template is not None:
                schedule = sched_template.assign_parameters({amp_param: amp_val})
            for control_state in range(2):
                exp = CycledRepeatedCRRxScan(physical_qubits, control_state, rcr_type,
                                             schedule=schedule, angles=angles, backend=backend)
                exp.set_experiment_options(
                    gate_params={'crp': [amp_val], 'crm': [amp_val]}
                )
                exp.extra_metadata['cr_amp'] = amp_val
                experiments.append(exp)

        analyses = [exp.analysis for exp in experiments]
        super().__init__(experiments, backend=backend,
                         analysis=CycledRepeatedCRRxOffsetAmpScanAnalysis(analyses))


class CycledRepeatedCRRxOffsetAmpScanAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

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
            AnalysisResultData(name='angle', value=angle_opt)
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

        return analysis_results, figures


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
        cal_parameter_name: list[str] = ['cr_amp', 'angle'],
        schedule_name: list[str] = ['cr', 'cx_offset_rx'],
        current_cal_group: list[str] = ['default', 'default'],
        angles: Optional[Sequence[float]] = None,
        auto_update: bool = True
    ):
        current_cr_amp = calibrations.get_parameter_value(cal_parameter_name[0], physical_qubits,
                                                          schedule_name[0],
                                                          group=current_cal_group[0])
        cr_amps = np.linspace(current_cr_amp - 0.05, current_cr_amp + 0.05, 4)
        rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)

        super().__init__(
            calibrations,
            physical_qubits,
            rcr_type,
            cr_amps,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            angles=angles
        )
        self.set_experiment_options(
            calibration_qubit_index={(self._param_name[1], self._sched_name[1]): [1]}
        )

        self._gate_name = QutritQubitCXGate.of_type(rcr_type).gate_name
        assign_keys = (
            (self._param_name[0], self.physical_qubits, self._sched_name[0]),
            (self._param_name[1], self.physical_qubits[1:], self._sched_name[1])
        )
        self._angle_param = Parameter('angle')
        self._schedules = []
        for aval in cr_amps:
            assign_params = dict(zip(assign_keys, [aval, self._angle_param]))
            self._schedules.append(
                get_qutrit_qubit_composite_gate(self._gate_name, physical_qubits, calibrations,
                                                target=backend.target, assign_params=assign_params)
            )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        iamp = circuit.metadata['composite_index'][0]
        angle_val = circuit.metadata['composite_metadata'][0]['xval']
        sched = self._schedules[iamp].assign_parameters({self._angle_param: angle_val},
                                                        inplace=False)
        circuit.add_calibration(self._gate_name, self.physical_qubits, sched, params=[angle_val])

    def update_calibrations(self, experiment_data: ExperimentData):
        qubit_lists = [self.physical_qubits, [self.physical_qubits[1]]]
        for pname, sname, qubits in zip(self._param_name, self._sched_name, qubit_lists):
            value = BaseUpdater.get_value(experiment_data, pname)
            param_value = ParameterValue(
                value=value,
                date_time=BaseUpdater._time_stamp(experiment_data),
                group=self.experiment_options.group,
                exp_id=experiment_data.experiment_id,
            )
            self._cals.add_parameter_value(param_value, pname, qubits, schedule=sname)
