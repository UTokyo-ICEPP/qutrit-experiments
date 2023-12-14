from typing import Any, Dict, List, Tuple, Iterable, Optional, Union, Sequence
import warnings
from uncertainties import ufloat, correlated_values, unumpy as unp
import numpy as np
import numpy.polynomial as poly
from matplotlib.figure import Figure
import scipy.optimize as sciopt
import lmfit

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
from qiskit_experiments.framework import (Options, BaseAnalysis, BaseExperiment, ExperimentData,
                                          AnalysisResultData)
from qiskit_experiments.framework.matplotlib import default_figure_canvas
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.curve_analysis.utils import eval_with_uncertainties, convert_lmfit_result
from qiskit_experiments.database_service import ExperimentEntryNotFound
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...common.framework_overrides import CompoundAnalysis, CompositeAnalysis, BatchExperiment
from ...common.linked_curve_analysis import LinkedCurveAnalysis
from ...common.util import sparse_poly_fitfunc, PolynomialOrder
from ..cr_rabi import cr_rabi_init
from ..hamiltonian_tomography import HamiltonianTomography, HamiltonianTomographyScan
from ..unitary_tomography import UnitaryTomography

twopi = 2. * np.pi


class QutritCR01Hamiltonian(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        widths: Optional[Iterable[float]] = None,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for control_state in range(2):
            exp = HamiltonianTomography(physical_qubits, schedule,
                                        rabi_init=cr_rabi_init(control_state), widths=widths,
                                        backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=QutritCR01HamiltonianAnalysis(analyses))


class QutritCR01HamiltonianAnalysis(CompoundAnalysis):
    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Compute the rotary frequency to align the Z/X ratios."""
        component_index = experiment_data.metadata["component_child_index"]

        omega = [2. * experiment_data.child_data(idx).analysis_results('hamiltonian_components').value
                 for idx in component_index]

        # (omega_0[0] + omega_r) / omega_0[2] = (omega_1[0] + omega_r) / omega_1[2]
        omega_r = (omega[0][2] * omega[1][0] - omega[1][2] * omega[0][0]) / (omega[1][2] - omega[0][2])

        analysis_results.append(AnalysisResultData('rotary_omega_x', value=omega_r))

        return analysis_results, figures


class QutritRCRRotaryFreqCal(BaseCalibrationExperiment, QutritCR01Hamiltonian):
    """Experiment to determine the rotary tone amplitude for repeated CR."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'rotary_omega_x',
        cal_parameter_name: str = 'rotary_omega_x',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        widths: Optional[Iterable[float]] = None
    ):
        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.,
            'cr_stark_amp': 0.,
            'counter_amp': 0.
        }
        schedule = calibrations.get_schedule('cr', physical_qubits,
                                             assign_params=assign_params, group=group)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            schedule=self._sched_name,
            group=self.experiment_options.group
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Find the rotary X frequency that aligns the Z-X angle of 0 and 1 blocks."""
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        BaseUpdater.update(self._cals, experiment_data, self._param_name, self._sched_name,
                           result_index=result_index, group=group, fit_parameter=self._param_name)


class QutritRCRRotaryAmp(BatchExperiment):
    """BatchExperiment of UnitaryTomographies scanning the rotary amp versus phi."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(-0.05, 0.05, 12)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        amp_param_name: str = 'counter_amp',
        angle_param_name: str = 'counter_phase',
        amplitudes: Optional[Sequence[float]] = None,
        repetitions: Optional[Sequence[int]] = None,
        backend: Optional[Backend] = None
    ):
        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        amp_param = schedule.get_parameters(amp_param_name)[0]
        angle_param = schedule.get_parameters(angle_param_name)[0]

        pre_circuit = QuantumCircuit(2)
        pre_circuit.x(0)

        experiments = []
        analyses = []

        for iexp, amplitude in enumerate(amplitudes_exp):
            angle = 0. if amplitude > 0. else np.pi
            assign_params = {amp_param: np.abs(amplitude), angle_param: angle}
            sched = schedule.assign_parameters(assign_params, inplace=False)

            exp = UnitaryTomography(physical_qubits, sched, repetitions=repetitions,
                                    measured_logical_qubit=1, backend=backend)
            exp.set_experiment_options(pre_circuit=pre_circuit)
            exp.extra_metadata['rotary_amp'] = amplitude
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, backend=backend,
                         analysis=QutritRCRRotaryAmpAnalysis(analyses))


class QutritRCRRotaryAmpAnalysis(CompoundAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.plot = True
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Compute the rotary frequency to align the Z/X ratios."""
        component_index = experiment_data.metadata["component_child_index"]

        amplitudes = []
        theta_y = []
        theta_y_err = []
        for idx in component_index:
            child_data = experiment_data.child_data(idx)
            amplitudes.append(child_data.metadata['rotary_amp'])
            popt = next(res.value.ufloat_params for res in child_data.analysis_results()
                        if res.name.startswith(PARAMS_ENTRY_PREFIX))
            val = popt['theta'] * unp.sin(popt['psi']) * unp.sin(popt['phi'])
            theta_y.append(val.n)
            theta_y_err.append(val.std_dev)

        amplitudes = np.array(amplitudes)
        theta_y = np.array(theta_y)
        theta_y_err = np.array(theta_y_err)

        if np.all(np.isfinite(theta_y_err)):
            weights = 1. / theta_y_err
        else:
            weights = None

        model = lmfit.models.LinearModel()
        slope_guess = (theta_y[-1] - theta_y[0]) / (amplitudes[-1] - amplitudes[0])
        intercept_guess = theta_y[-1] - slope_guess * amplitudes[-1]
        initial_params = {'slope': slope_guess, 'intercept': intercept_guess}
        result = model.fit(theta_y, weights=weights, x=amplitudes, **initial_params)
        fit_data = convert_lmfit_result(result, [model], amplitudes, theta_y)

        analysis_results.append(
            AnalysisResultData(name=PARAMS_ENTRY_PREFIX + type(self).__name__, value=fit_data)
        )

        # Rotary amplitude to make theta_y = 0
        params = fit_data.ufloat_params
        rotary_amp = -params['intercept'] / params['slope']
        if rotary_amp.n > 0.:
            rotary_angle = 0.
        else:
            rotary_amp *= -1.
            rotary_angle = np.pi

        analysis_results += [
            AnalysisResultData(name='rotary_amp', value=rotary_amp),
            AnalysisResultData(name='rotary_angle', value=ufloat(rotary_angle, 0.))
        ]

        if self.options.plot:
            plotter = CurvePlotter(MplDrawer())
            plotter.set_figure_options(
                xlabel='Rotary amplitude',
                ylabel='Gate Y component'
            )
            plotter.set_series_data(
                model._name,
                x_formatted=amplitudes,
                y_formatted=unp.nominal_values(theta_y),
                y_formatted_err=unp.std_devs(theta_y)
            )
            if fit_data.success:
                plotter.set_supplementary_data(fit_red_chi=fit_data.reduced_chisq)
                x_interp = np.linspace(np.min(amplitudes), np.max(amplitudes), 100)
                y_data_with_uncertainty = eval_with_uncertainties(
                    x=x_interp,
                    model=model,
                    params=fit_data.ufloat_params
                )
                y_interp = unp.nominal_values(y_data_with_uncertainty)
                # Add fit line data
                plotter.set_series_data(
                    model._name,
                    x_interp=x_interp,
                    y_interp=y_interp,
                )
                if fit_data.covar is not None:
                    # Add confidence interval data
                    y_interp_err = unp.std_devs(y_data_with_uncertainty)
                    if np.isfinite(y_interp_err).all():
                        plotter.set_series_data(
                            model._name,
                            y_interp_err=y_interp_err,
                        )

            figures.append(plotter.figure())

        return analysis_results, figures


class QutritRCRRotaryAmpCal(BaseCalibrationExperiment, QutritRCRRotaryAmp):
    """Experiment to determine the rotary tone amplitude for repeated CR."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: List[str] = ['counter_amp', 'counter_phase'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Sequence[float]] = None,
        repetitions: Optional[Sequence[int]] = None
    ):
        physical_qubits = tuple(physical_qubits)

        assign_params = {
            (cal_parameter_name[0], physical_qubits, schedule_name): Parameter('counter_amp'),
            (cal_parameter_name[1], physical_qubits, schedule_name): Parameter('counter_phase')
        }
        schedule = calibrations.get_schedule('rcr', physical_qubits,
                                             assign_params=assign_params, group=group)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            amp_param_name=cal_parameter_name[0],
            angle_param_name=cal_parameter_name[1],
            amplitudes=amplitudes,
            repetitions=repetitions
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group
            ) for pname in self._param_name
        ]
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Find the rotary X frequency that aligns the Z-X angle of 0 and 1 blocks."""
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        for pname, fit_pname in zip(self._param_name, ['rotary_amp', 'rotary_angle']):
            BaseUpdater.update(self._cals, experiment_data, pname, self._sched_name,
                               result_index=result_index, group=group, fit_parameter=fit_pname)
