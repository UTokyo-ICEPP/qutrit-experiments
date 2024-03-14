"""UT scan over target Stark amplitudes to minimize the Iz component of the unitary."""
from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
import scipy.optimize as sciopt
from uncertainties import correlated_values, ufloat, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from .qutrit_qubit_tomography import QutritQubitTomographyScan, QutritQubitTomographyScanAnalysis


class QutritCRTargetStarkCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """UT scan over target Stark amplitudes to minimize the Iz component for the given set of
    control blocks."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'counter_stark_amp',
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        control_states: tuple[int, ...] = (0, 1, 2),
        auto_update: bool = True
    ):
        counter_stark_amp = Parameter(cal_parameter_name)
        assign_params = {cal_parameter_name: counter_stark_amp}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        circuit = QuantumCircuit(2)
        circuit.append(Gate('cr', 2, [counter_stark_amp]), [0, 1])
        circuit.add_calibration('cr', physical_qubits, schedule, [counter_stark_amp])

        if amplitudes is None:
            amplitudes = np.linspace(0.005, 0.16, 6)

        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            param_name=cal_parameter_name,
            values=amplitudes,
            measure_preparations=measure_preparations,
            control_states=control_states,
            analysis_cls=QutritCRTargetStarkAnalysis,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        BaseUpdater.update(
            self._cals, experiment_data, self._param_name, schedule=self._sched_name,
            group=self.experiment_options.group, fit_parameter='counter_stark_amp'
        )


class QutritCRTargetStarkAnalysis(QutritQubitTomographyScanAnalysis):
    """Apply a quadratic fit to the observed theta_zs."""
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.figure_names.append('theta_iz')
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: list[AnalysisResultData],
        figures: list[Figure]
    ) -> tuple[list[AnalysisResultData], list[Figure]]:
        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        amplitudes = experiment_data.metadata['scan_values'][0]
        unitaries = next(res.value for res in analysis_results if res.name == 'unitary_parameters')
        theta_iz = np.mean(unitaries[..., 2], axis=1)
        theta_iz_n = unp.nominal_values(theta_iz)
        theta_iz_e = unp.std_devs(theta_iz)

        amp_sq = np.square(amplitudes)
        sigma_sq = np.square(theta_iz_e)

        def fun(params):
            return np.sum(np.square(params[0] * amp_sq + params[1] - theta_iz_n) / sigma_sq)

        jac_coeff = 2. * np.array([amp_sq, np.ones_like(amplitudes)])
        def jac(params):
            return np.sum(jac_coeff * (params[0] * amp_sq + params[1] - theta_iz_n)[None, :]
                           / sigma_sq[None, :],
                          axis=-1)

        hess_coeff = 2. * np.array(
            [[amp_sq ** 2, amp_sq],
             [amp_sq, np.ones_like(amplitudes)]]
        )
        def hess(params):
            return np.sum(hess_coeff * (params[0] * amp_sq + params[1] - theta_iz_n)[None, None, :]
                           / sigma_sq[None, None, :],
                          axis=-1)

        a_p0 = theta_iz_n[-1] / amp_sq[-1]
        result = sciopt.minimize(fun, (a_p0, 0.), jac=jac, hess=hess)

        if result.x[0] * result.x[1] > 0.:
            amp = ufloat(0., 0.)
        else:
            popt_ufloats = correlated_values(result.x, result.hess_inv * 2.)
            amp = unp.sqrt(-popt_ufloats[1] / popt_ufloats[0])[()]

        analysis_results.append(AnalysisResultData(name='counter_stark_amp', value=amp))

        if self.options.plot:
            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)
            plotter = CurvePlotter(MplDrawer())
            plotter.set_series_data(
                'theta_iz',
                x_formatted=amplitudes,
                y_formatted=theta_iz_n,
                y_formatted_err=unp.std_devs(theta_iz),
                x_interp=x_interp,
                y_interp=result.x[0] * np.square(x_interp) + result.x[1]
            )
            figures.append(plotter.figure())

        return analysis_results, figures



class QutritCRTargetStarkSingleControlCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """UT scan over target Stark amplitudes to minimize the z component of the unitary of a single
    control state block."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        control_state: int,
        backend: Optional[Backend] = None,
        cal_parameter_name: str = 'counter_stark_amp',
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        counter_stark_amp = Parameter(cal_parameter_name)
        assign_params = {cal_parameter_name: counter_stark_amp}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        circuit = QuantumCircuit(2)
        circuit.append(Gate('cr', 2, [counter_stark_amp]), [0, 1])
        circuit.add_calibration('cr', physical_qubits, schedule, [counter_stark_amp])

        if amplitudes is None:
            amplitudes = np.linspace(0.005, 0.16, 6)

        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            param_name=cal_parameter_name,
            values=amplitudes,
            measure_preparations=measure_preparations,
            control_states=(control_state,),
            analysis_cls=QutritCRTargetStarkAnalysis,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        BaseUpdater.update(
            self._cals, experiment_data, self._param_name, schedule=self._sched_name,
            group=self.experiment_options.group, fit_parameter='counter_stark_amp'
        )


class QutritCRControlStarkCal(BaseCalibrationExperiment, QutritQubitTomographyScan):
    """UT scan over control Stark amplitudes to minimize the zz & ζz components of the unitary."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: list[str] = ['cr_stark_amp', 'cr_stark_sign_phase'],
        schedule_name: str = 'cr',
        amplitudes: Optional[Sequence[float]] = None,
        measure_preparations: bool = True,
        auto_update: bool = True
    ):
        parameters = [Parameter(pname) for pname in cal_parameter_name]
        assign_params = {p.name: p for p in parameters}
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params)

        circuit = QuantumCircuit(2)
        circuit.append(Gate('cr', 2, parameters), [0, 1])
        circuit.add_calibration('cr', physical_qubits, schedule, parameters)

        if amplitudes is None:
            max_amp = 0.99 - calibrations.get_parameter_value('cr_amp', physical_qubits,
                                                              schedule_name)
            amplitudes = np.linspace(-max_amp, max_amp, 8)

        super().__init__(
            calibrations,
            physical_qubits,
            circuit,
            param_name=cal_parameter_name[0],
            values=amplitudes,
            measure_preparations=measure_preparations,
            backend=backend,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            angle_param_name=cal_parameter_name[1]
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        component_index = experiment_data.metadata['component_child_index']
        amplitudes = [experiment_data.child_data(idx).metadata[self._param_name[0]]
                      for idx in component_index]
        unitaries = experiment_data.analysis_results('unitary_parameters', block=False).value
        unitaries = unp.nominal_values(unitaries)
        theta_zs = np.einsum('oc,sck->sok',
                             np.linalg.inv([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                             unitaries)[:, 1:, 2]
        best_amp = amplitudes[np.argmin(np.sum(np.square(theta_zs), axis=-1))]
        phase = 0. if best_amp > 0. else np.pi
        for pname, value in zip(self._param_name, [best_amp, phase]):
            BaseUpdater.add_parameter_value(
                self._cals, experiment_data, value, pname, schedule=self._sched_name,
                group=self.experiment_options.group
            )

