from typing import Iterable, Optional, Sequence, Dict, Any
import warnings
import numpy as np
import numpy.polynomial as poly
import scipy.optimize as sciopt
from uncertainties import ufloat, unumpy as unp

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit_experiments.framework import Options, AnalysisResultData, ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX

from ...common.framework_overrides import BatchExperiment
from ...common.util import PolynomialOrder
from ..qutrit_cr_amplitude import QutritCRAmplitude
from .cr_multi_simul import CRMultiSimultaneousFitAnalysis

class QutritCRAmplitude02Sync(BatchExperiment):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        analysis_poly_order: PolynomialOrder = 5,
        backend: Optional[Backend] = None
    ):
        experiments = []
        analyses = []

        for control_state in [0, 2]:
            exp = QutritCRAmplitude(physical_qubits,
                                    control_state,
                                    schedule,
                                    amplitudes=amplitudes,
                                    widths=widths,
                                    time_unit=time_unit,
                                    analysis_poly_order=analysis_poly_order,
                                    backend=backend)
            experiments.append(exp)
            analyses.append(exp.analysis)

        analysis = QutritCRAmplitude02SyncAnalysis(analyses)
        super().__init__(experiments, backend=backend, analysis=analysis)


class QutritCRAmplitude02SyncAnalysis(CRMultiSimultaneousFitAnalysis):
    def _plot_curves(self, experiment_data):
        self.plotter.set_figure_options(
            xlabel='CR amplitude',
            ylabel='Rabi frequency',
        )

        component_index = experiment_data.metadata["component_child_index"]
        child_data = [experiment_data.child_data(component_index[ichld]) for ichld in range(2)]

        for analysis, data in zip(self._analyses, child_data):
            fit_result = data.analysis_results(PARAMS_ENTRY_PREFIX + analysis.name).value

            popt = fit_result.ufloat_params
            popt_individual = analysis.get_individual_fit_results(data)

            try:
                control_state = data.metadata['control_state']
            except KeyError:
                # backward compat
                control_state = analysis.options.extra['control_state']

            cr_amps = np.array(list(grandchild_data.metadata['cr_amp'] for grandchild_data in data.child_data()))

            analysis.draw_linked_param('freq',
                                       analysis.linked_params['freq'],
                                       popt,
                                       popt_individual,
                                       'cr_amp',
                                       cr_amps,
                                       plotter=self.plotter,
                                       name=fr'Ctrl=$|{control_state}\rangle$',
                                       indiv_result_label=None,
                                       simul_result_label=None)

    def _solve(self, fit_results: Dict[int, curve.CurveFitResult], metadata: Dict[int, Dict[str, Any]]):
        poly_orders = {}
        popts = {}
        xmin = 1.
        xmax = 0.
        for control_state, analysis in zip([0, 2], self._analyses):
            poly_orders[control_state] = analysis.poly_order
            popts[control_state] = fit_results[control_state].ufloat_params

            xvals = list(d['cr_amp'] for d in metadata[control_state]['component_metadata'])
            xmin = min(xmin, min(xvals))
            xmax = max(xmax, max(xvals))

        order = max(poly_order.order for poly_order in poly_orders.values())
        freq_coeffs = np.array(list(ufloat(0., 0.) for _ in range(order + 1)))
        for power in poly_orders[0].powers:
            freq_coeffs[power] += popts[0][f'freq_c{power}']
        for power in poly_orders[2].powers:
            freq_coeffs[power] -= popts[2][f'freq_c{power}']

        f = lambda x: unp.nominal_values(poly.polynomial.polyval(x, freq_coeffs))

        cr_amp = None

        # Start from the scanned range and find a point where f changes sign, then solve f=0 using
        # that point as the initial guess

        x0 = None
        while x0 is None:
            finex = np.arange(xmin, xmax, 0.01)

            sign_changes = np.nonzero((f(finex[:-1]) * f(finex[1:])) < 0.)[0]
            if len(sign_changes) > 0:
                # Choose the largest of such xs
                x0 = finex[sign_changes[-1]]
            else:
                xmin = max(xmin - 0.05, 0.)
                xmax = min(xmax + 0.05, 1.)
                if xmin == 0. and xmax == 1.:
                    break
        else:
            x1 = x0 - 0.01

            fprime_coeffs = poly.polynomial.polyder(freq_coeffs)
            fprime2_coeffs = poly.polynomial.polyder(freq_coeffs, m=2)

            fprime = lambda x: unp.nominal_values(poly.polynomial.polyval(x, fprime_coeffs))
            fprime2 = lambda x: unp.nominal_values(poly.polynomial.polyval(x, fprime2_coeffs))

            sol = sciopt.root_scalar(f, fprime=fprime, fprime2=fprime2, x0=x0, x1=x1)
            if sol.converged:
                nominal = sol.root
                f_1sigma = lambda x: unp.std_devs(poly.polynomial.polyval(x, freq_coeffs))
                f_plus1sigma = lambda x: f(x) + f_1sigma(x)
                f_minus1sigma = lambda x: f(x) - f_1sigma(x)
                sol_plus1sigma = sciopt.root_scalar(f_plus1sigma, x0=x0, x1=x1)
                sol_minus1sigma = sciopt.root_scalar(f_minus1sigma, x0=x0, x1=x1)
                stddev = max(abs(sol_plus1sigma.root - nominal), abs(sol_minus1sigma.root - nominal))
                cr_amp = ufloat(nominal, stddev)

        if cr_amp is None or np.isclose(cr_amp.n, 0.):
            warnings.warn(
                f"Failed to find a value for cr_amp",
                UserWarning,
            )

            return []

        analysis_results = [
            AnalysisResultData(name='cr_amp', value=cr_amp)
        ]

        return analysis_results


class QutritCXCRAmplitudeCal(BaseCalibrationExperiment, QutritCRAmplitude02Sync):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: str = 'cr_amp',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Iterable[float]] = None,
        counter_amplitude: Optional[float] = None,
        rabi_schedule: str = 'cr',
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        analysis_poly_order: PolynomialOrder = 5
    ):
        physical_qubits = tuple(physical_qubits)

        params_values = [
            (cal_parameter_name, Parameter('cr_amp')),
            ('cr_stark_amp', 0.),
            ('cr_width', Parameter('cr_width')),
            ('cr_margin', 0.),
            ('counter_stark_amp', 0.)
        ]
        if counter_amplitude is None:
            params_values.append(('counter_amp', 0.))
        else:
            params_values.append(('counter_amp', counter_amplitude))
            params_values.append(('counter_phase', 0.))

        if rabi_schedule == 'cr':
            schedule = calibrations.get_schedule(
                'cr', physical_qubits, assign_params=dict(params_values), group=group
            )
        else:
            assign_params = {(pname, physical_qubits, schedule_name): value
                             for pname, value in params_values}
            assign_params[('rx_amp', physical_qubits, 'rx')] = 0.

            qutrit_cx = calibrations.get_schedule(
                'qutrit_cx', physical_qubits, assign_params=assign_params, group=group
            )
            if rabi_schedule == 'cx':
                schedule = qutrit_cx
            elif rabi_schedule == 'cxcx':
                with pulse.build(name='cxcx') as schedule:
                    pulse.call(qutrit_cx)
                    pulse.call(qutrit_cx)
            else:
                raise NotImplementedError(rabi_schedule)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            amplitudes=amplitudes,
            widths=widths,
            time_unit=time_unit,
            analysis_poly_order=analysis_poly_order
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = self._cals.get_parameter_value(
            self._param_name,
            self.physical_qubits,
            schedule=self._sched_name,
            group=self.experiment_options.group,
        )

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        BaseUpdater.update(self._cals, experiment_data, self._param_name, self._sched_name,
                           result_index=result_index, group=group, fit_parameter=self._param_name)
