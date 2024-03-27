"""UT scan over target Stark amplitudes to minimize the Iz component of the unitary."""
from collections.abc import Sequence
from typing import Optional
from matplotlib.figure import Figure
import numpy as np
from scipy.optimize import curve_fit
from uncertainties import correlated_values, ufloat, unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.providers import Backend, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import AnalysisResultData, ExperimentData, Options
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...gates import CrossResonanceGate
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
        circuit.append(CrossResonanceGate(params=[counter_stark_amp]), [0, 1])
        circuit.add_calibration(CrossResonanceGate.gate_name, physical_qubits, schedule,
                                [counter_stark_amp])

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
        theta_iz = np.mean([params[:, 2] for params in unitaries.values()], axis=0)
        theta_iz_n = unp.nominal_values(theta_iz)
        theta_iz_e = unp.std_devs(theta_iz)

        def curve(x2, curvature, offset):
            return curvature * x2 + offset

        ampsq = np.square(amplitudes)
        p0 = (theta_iz[-1].n / ampsq[-1], 0.)
        popt, pcov = curve_fit(curve, ampsq, theta_iz_n, sigma=theta_iz_e, p0=p0)
        
        if popt[0] * popt[1] > 0.:
            amp = ufloat(0., 0.)
        else:
            popt_ufloats = correlated_values(popt, pcov)
            amp = unp.sqrt(-popt_ufloats[1] / popt_ufloats[0])[()]

        analysis_results.append(AnalysisResultData(name='counter_stark_amp', value=amp))

        if self.options.plot:
            x_interp = np.linspace(amplitudes[0], amplitudes[-1], 100)
            plotter = CurvePlotter(MplDrawer())
            plotter.set_series_data(
                'theta_iz',
                x_formatted=amplitudes,
                y_formatted=unp.nominal_values(theta_iz),
                y_formatted_err=unp.std_devs(theta_iz),
                x_interp=x_interp,
                y_interp=curve(x_interp ** 2, *popt)
            )
            figures.append(plotter.figure())

        return analysis_results, figures


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
        circuit.append(CrossResonanceGate(params=parameters), [0, 1])
        circuit.add_calibration(CrossResonanceGate.gate_name, physical_qubits, schedule, parameters)

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

