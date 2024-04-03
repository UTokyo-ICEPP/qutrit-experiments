from collections.abc import Sequence
from threading import Lock
from typing import Optional, Union
from matplotlib.figure import Figure
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options

from ...calibrations import get_qutrit_qubit_composite_gate
from ...gates import RCRGate
from ..qutrit_qubit.qutrit_qubit_tomography import (QutritQubitTomographyScan,
                                                    QutritQubitTomographyScanAnalysis)


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
        options.cx_sign = 0
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

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        fit_params = next(res.value for res in analysis_results if res.name == 'simul_fit_params')
        nonpart_state = experiment_data.metadata['control_states'][0]
        slope = (fit_params[1][0] - fit_params[nonpart_state][0])
        intercept = (fit_params[1][1] - fit_params[nonpart_state][1])
        target_angle = np.pi / 2. * self.options.cx_sign
        pi_amp = (target_angle - intercept) / slope
        analysis_results.append(
            AnalysisResultData(name='pi_amp', value=pi_amp)
        )
        return analysis_results, figures


class CRRoughAmplitudeCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """Calibration of CR amplitude given the width.

    Rx angle difference between block 1 and block rcr_type will be set to cx_sign * pi/2.
    If two cal_parameter_names are given, the second one will be for the offset Rx angle, which
    will be set to cancel the angle of block rcr_type, but this calibration will be rendered
    meaningless once the rotary tone is introduced.
    """
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
        cal_parameter_name: Union[str, list[str]] = 'cr_amp',
        schedule_name: Union[str, list[str]] = 'cr',
        auto_update: bool = True,
        amplitudes: Optional[Sequence[float]] = None
    ):
        if isinstance(cal_parameter_name, str):
            cr_amp_param_name = cal_parameter_name
            cr_amp_sched_name = schedule_name
        else:
            cr_amp_param_name = cal_parameter_name[0]
            cr_amp_sched_name = schedule_name[0]
        if amplitudes is None:
            current = calibrations.get_parameter_value(cr_amp_param_name, physical_qubits,
                                                       cr_amp_sched_name)
            amplitudes = np.linspace(current - 0.2, current + 0.05, 6)

        rcr_type = calibrations.get_parameter_value('rcr_type', physical_qubits)
        gate = RCRGate.of_type(rcr_type)(params=[Parameter('amp')])
        super().__init__(
            calibrations,
            physical_qubits,
            gate,
            'amp',
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            values=amplitudes,
            control_states=(rcr_type, 1),
            analysis_cls=RepeatedCRAmplitudeAnalysis
        )
        if not isinstance(cal_parameter_name, str):
            self.set_experiment_options(
                calibration_qubit_index={(self._param_name[1], self._sched_name[1]): [1]}
            )

        self._gate_name = gate.name
        assign_key = (cr_amp_param_name, self.physical_qubits, cr_amp_sched_name)
        self._schedules = [
            get_qutrit_qubit_composite_gate(self._gate_name, physical_qubits, calibrations,
                                            target=backend.target, assign_params={assign_key: aval})
            for aval in amplitudes
        ]

        self.analysis.set_options(
            cx_sign=calibrations.get_parameter_value('qutrit_qubit_cx_sign', physical_qubits)
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        iamp = circuit.metadata['composite_index'][0]
        circuit.add_calibration(self._gate_name, self.physical_qubits, self._schedules[iamp],
                                params=[self.experiment_options.parameter_values[0][iamp]])

    def update_calibrations(self, experiment_data: ExperimentData):
        if isinstance(self._param_name, str):
            cr_amp_param_name = self._param_name
            cr_amp_sched_name = self._sched_name
            rx_angle_param_name = rx_angle_sched_name = None
        else:
            cr_amp_param_name = self._param_name[0]
            cr_amp_sched_name = self._sched_name[0]
            rx_angle_param_name = self._sched_name[1]
            rx_angle_sched_name = self._sched_name[1]

        new_amp = experiment_data.analysis_results('pi_amp', block=False).value.n
        if new_amp > 0.95:
            return

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, new_amp, cr_amp_param_name,
            schedule=cr_amp_sched_name, group=self.experiment_options.group
        )

        if rx_angle_param_name:
            fit_params = experiment_data.analysis_results('simul_fit_params', block=False).value
            nonpart_state = experiment_data.metadata['control_states'][0]
            theta_0 = fit_params[nonpart_state][0].n * new_amp + fit_params[nonpart_state][1].n
            param_value = ParameterValue(
                value=-theta_0,
                date_time=BaseUpdater._time_stamp(experiment_data),
                group=self.experiment_options.group,
                exp_id=experiment_data.experiment_id,
            )
            self._cals.add_parameter_value(param_value, rx_angle_param_name, self.physical_qubits[1],
                                           schedule=rx_angle_sched_name)
