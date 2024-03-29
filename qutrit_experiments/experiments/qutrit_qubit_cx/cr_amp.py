from collections.abc import Sequence
from threading import Lock
from typing import Optional
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.calibration_management import (BaseCalibrationExperiment, Calibrations,
                                                       ParameterValue)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import ExperimentData, Options

from ...gates import CrossResonanceGate, RCRGate
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


class CRRoughAmplitudeCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """Calibration of CR amplitude given the width.

    Rx angle difference between block 1 and block rcr_type will be set to cx_sign * pi/2.
    Offset Rx angle is set to cancel the angle of block rcr_type, but this calibration will be
    rendered meaningless once the rotary tone is introduced.
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
        cal_parameter_name: list[str] = ['cr_amp', 'angle'],
        schedule_name: list[str] = ['cr', 'cx_offset_rx'],
        auto_update: bool = True,
        amplitudes: Optional[Sequence[float]] = None
    ):
        if amplitudes is None:
            current = calibrations.get_parameter_value(cal_parameter_name[0], physical_qubits,
                                                       schedule_name[0])
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
        self.set_experiment_options(
            calibration_qubit_index={(self._param_name[1], self._sched_name[1]): [1]}
        )

        self._gate_name = gate.name
        assign_keys = [
            ('freq', self.physical_qubits[:1], 'x12'),
            (self._param_name[0], self.physical_qubits, self._sched_name[0])
        ]
        freq = (calibrations.get_parameter_value('f12', physical_qubits[0])
                - backend.qubit_properties(physical_qubits[0]).frequency)
        self._schedules = [
            calibrations.get_schedule(self._gate_name, physical_qubits,
                                      assign_params=dict(zip(assign_keys, [freq, aval])))
            for aval in amplitudes
        ]

    def _attach_calibrations(self, circuit: QuantumCircuit):
        iamp = circuit.metadata['composite_index'][0]
        circuit.add_calibration(self._gate_name, self.physical_qubits, self._schedules[iamp],
                                params=[self.experiment_options.parameter_values[0][iamp]])

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
        self._cals.add_parameter_value(param_value, self._param_name[1], self.physical_qubits[1],
                                       schedule=self._sched_name[1])
