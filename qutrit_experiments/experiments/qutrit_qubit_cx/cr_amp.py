from collections.abc import Sequence
from threading import Lock
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ...util.polynomial import PolynomialOrder
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)
from .util import RCRType, make_rcr_circuit


class RepeatedCRAmplitude(QutritQubitTomographyScan):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        amplitudes = np.linspace(0., 0.8, 9)
        options.parameter_values = [amplitudes]
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        cr_schedule: ScheduleBlock,
        rcr_type: RCRType,
        amp_param_name: str = 'cr_amp',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        backend: Optional[Backend] = None
    ):
        if amplitudes is None:
            amplitudes = self._default_experiment_options().parameter_values[0]

        super().__init__(physical_qubits,
                         make_rcr_circuit(physical_qubits, cr_schedule, rcr_type),
                         amp_param_name, amplitudes, measure_preparations=measure_preparations,
                         control_states=(int(rcr_type), 1), backend=backend,
                         analysis_cls=RepeatedCRAmplitudeAnalysis)


class RepeatedCRAmplitudeAnalysis(QutritQubitTomographyScanAnalysis):
    """Analysis for RepeatedCRAmplitude.
    
    Simultaneous fit model is
    [x, y, z] = [ax*A + bx, ay*A + by, az*A + bz]
    although the amplitude dependency shouldn't really be linear. Should be OK since the experiment
    is used to scan a narrow range of amplitudes.
    """
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.simul_fit = True
        return options
    
    _compile_lock = Lock()
    _fit_functions_cache = {}

    @classmethod
    def unitary_params(cls, fit_params: np.ndarray, aval: np.ndarray, npmod=np):
        if npmod is np:
            aval = np.asarray(aval)
        slopes = fit_params[::2]
        intercepts = fit_params[1::2]
        return aval[..., None] * slopes + intercepts
    
    def _get_p0s(self, norm_xvals: np.ndarray, unitary_params: np.ndarray):
        p0s = []
        for axis in range(3):
            params = unp.nominal_values(unitary_params[:, axis])
            slope = np.diff(params[[0, -1]])[0] / np.diff(norm_xvals[[0, -1]])[0]
            intercept = params[0] - slope * norm_xvals[0]
            p0s.extend([slope, intercept])
        return np.array([p0s])

    def _postprocess_params(self, upopt: np.ndarray, norm: float):
        upopt[[0, 2, 4]] /= norm

    
class CRRoughAmplitudeCal(BaseCalibrationExperiment, RepeatedCRAmplitude):
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
        schedule_name: list[str] = ['cr', None],
        auto_update: bool = True,
        amplitudes: Optional[Sequence[float]] = None
    ):
        cr_amp = Parameter('cr_amp')
        assign_params = {cal_parameter_name[0]: cr_amp}
        schedule = calibrations.get_schedule(schedule_name[0], physical_qubits,
                                             assign_params=assign_params)

        if amplitudes is None:
            current = calibrations.get_parameter_value(cal_parameter_name[0], physical_qubits,
                                                       schedule_name[0])
            amplitudes = np.linspace(current - 0.2, current + 0.05, 6)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            RCRType(calibrations.get_parameter_value('rcr_type', physical_qubits)),
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            amp_param_name='cr_amp',
            amplitudes=amplitudes
        )
        self.set_experiment_options(calibration_qubit_index={(self._param_name[1], None): [1]})

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
        nonpart_state = experiment_data.metadata['control_states'][0]
        slope = (fit_params[1][0] - fit_params[nonpart_state][0]).n
        intercept = (fit_params[1][1] - fit_params[nonpart_state][1]).n
        cx_sign = self._cals.get_parameter_value('qutrit_qubit_cx_sign', self.physical_qubits)
        target_angle = np.pi / 2. * cx_sign
        new_amp = (target_angle - intercept) / slope
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_amp, self._param_name[0], schedule=self._sched_name[0],
            group=self.experiment_options.group
        )

        theta_0 = fit_params[nonpart_state][0].n * new_amp + fit_params[nonpart_state][1].n
        param_value = ParameterValue(
            value=-theta_0,
            date_time=BaseUpdater._time_stamp(experiment_data),
            group=self.experiment_options.group,
            exp_id=experiment_data.experiment_id,
        )
        self._cals.add_parameter_value(param_value, self._param_name[1], self.physical_qubits[1])
