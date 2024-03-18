from collections.abc import Sequence
import logging
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
import scipy.optimize as sciopt
from uncertainties import correlated_values, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ...util.pulse_area import gs_effective_duration, rabi_cycles_per_area
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, get_cr_schedules, make_crcr_circuit, make_rcr_circuit

twopi = 2. * np.pi
logger = logging.getLogger(__name__)


class RepeatedCRRotaryAmplitude(QutritQubitTomographyScan):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.01, 0.01, 12)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedule: ScheduleBlock,
        rcr_type: RCRType,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        control_states = list(range(3))
        control_states.remove(int(rcr_type))

        super().__init__(physical_qubits, make_rcr_circuit(physical_qubits, cr_schedule, rcr_type),
                         amp_param_name, amplitudes, angle_param_name=angle_param_name,
                         measure_preparations=measure_preparations, control_states=control_states,
                         backend=backend, analysis_cls=MinimumYZRotaryAmplitudeAnalysis)


class CycledRepeatedCRRotaryAmplitude(QutritQubitTomographyScan):
    """BatchExperiment of RepeatedCRTomography scanning the rotary amplitude."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.parameter_values = [np.linspace(-0.01, 0.01, 12)]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedules: tuple[ScheduleBlock, ScheduleBlock],
        rcr_type: RCRType,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_sign_angle',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits,
                         make_crcr_circuit(physical_qubits, cr_schedules, None, rcr_type),
                         amp_param_name, amplitudes, angle_param_name=angle_param_name,
                         measure_preparations=measure_preparations, backend=backend,
                         analysis_cls=MinimumYZRotaryAmplitudeAnalysis)


class MinimumYZRotaryAmplitudeAnalysis(QutritQubitTomographyScanAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.chi2_per_block_cutoff = 24.
        options.unitary_parameter_ylims = {'Y': (-0.2, 0.2), 'Z': (-0.2, 0.2)}
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        amplitudes = np.array(experiment_data.metadata['scan_values'][0])
        u_params = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        yz_n = np.array([unp.nominal_values(params[:, [1, 2]]) for params in u_params.values()])
        yz_e = np.array([unp.std_devs(params[:, [1, 2]]) for params in u_params.values()])

        # Cut on chi2 mean over states for each xval
        chisq = next(res for res in analysis_results if res.name == 'chisq').value
        good_fit = np.mean(list(chisq.values()), axis=0) < self.options.chi2_per_block_cutoff
        if not np.any(good_fit):
            logger.warning('No rotary value had sum of chi2 per block less than %f',
                           self.options.chi2_per_block_cutoff)
            good_fit = np.ones_like(amplitudes, dtype=bool)

        # Any amplitude value where y and z are consistent with zero for all control states?
        zero_consistent = np.all(np.abs(yz_n) < yz_e, axis=(0, 2))
                                 
        filter = good_fit & zero_consistent
        if np.any(filter):
            # There are points where y and z are consistent with zero -> choose one with the lowest
            # amp
            iamp = np.argmin(np.abs(amplitudes[filter]))
        else:
            # Pick the rotary value with the smallest y^2 + z^2
            filter = good_fit
            iamp = np.argmin(np.sum(np.square(yz_n[:, filter]), axis=(0, 2)))

        analysis_results.append(
            AnalysisResultData(name='rotary_amp', value=amplitudes[filter][iamp])
        )
        return analysis_results, figures


class RepeatedCRRotaryAmplitudeCal(BaseCalibrationExperiment, RepeatedCRRotaryAmplitude):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['counter_amp', 'counter_sign_angle'],
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[float] = None,
        measure_preparations: bool = True,
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
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amp_param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes,
            measure_preparations=measure_preparations
        )

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


class CycledRepeatedCRRotaryAmplitudeCal(BaseCalibrationExperiment,
                                         CycledRepeatedCRRotaryAmplitude):
    """Calibration experiment for CycledRepeatedCRRotaryAmplitude. Additionally calibrates offset Rx
    angle."""
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
        cal_parameter_name: list[str] = ['counter_amp', 'counter_sign_angle', 'qutrit_qubit_cx_offsetrx'],
        schedule_name: list[str] = ['cr', 'cr', None],
        amplitudes: Optional[Sequence[float]] = None,
        width: Optional[float] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        if width is not None:
            assign_params = {'width': width}
        else:
            assign_params = None
        cr_schedules = get_cr_schedules(calibrations, physical_qubits,
                                        free_parameters=cal_parameter_name[:2],
                                        assign_params=assign_params)

        super().__init__(
            calibrations,
            physical_qubits,
            cr_schedules,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amp_param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes,
            measure_preparations=measure_preparations
        )
        self.set_experiment_options(calibration_qubit_index={(self._param_name[2], None): [1]})

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        amplitude = experiment_data.analysis_results('rotary_amp', block=False).value
        angle = 0.
        if amplitude < 0.:
            amplitude *= -1.
            angle = np.pi

        for pname, value in zip(self._param_name[:2], [amplitude, angle]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name[0],
                group=self.experiment_options.group
            )

        selected_amp_idx = int(np.argmin(
            np.abs(experiment_data.metadata['scan_values'][0] - amplitude)
        ))
        unitaries = experiment_data.analysis_results('unitary_parameters', block=False).value
        # X angle of control=0
        rx_angle = -unitaries[selected_amp_idx, 0, 0].n
        param_value = ParameterValue(
            value=rx_angle,
            date_time=BaseUpdater._time_stamp(experiment_data),
            group=self.experiment_options.group,
            exp_id=experiment_data.experiment_id,
        )
        self._cals.add_parameter_value(param_value, self._param_name[2], self.physical_qubits[1])
