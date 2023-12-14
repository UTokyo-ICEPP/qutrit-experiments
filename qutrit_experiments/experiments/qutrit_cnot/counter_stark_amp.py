from typing import Sequence, Optional, Iterable, Tuple, Dict, List
import warnings
import numpy as np
import numpy.polynomial as poly
import scipy.optimize as sciopt
from uncertainties import ufloat, unumpy as unp

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.circuit import Parameter
from qiskit_experiments.framework import Options, AnalysisResultData, ExperimentData
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...common.util import sparse_poly_fitfunc, PolynomialOrder
from ..qutrit_cr_hamiltonian import QutritCRHamiltonianScan, QutritCRHamiltonianScanAnalysis
from ..hamiltonian_tomography import HamiltonianTomographyScan, HamiltonianTomographyScanAnalysis
from ..cr_rabi import cr_rabi_init


class QutritCXCounterStarkAmpCal(BaseCalibrationExperiment, QutritCRHamiltonianScan):
    """CR HT scan to find the Stark tone amplitude that cancels omega_Iz."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(0.01, 0.06, 6)
        options.update_cr_phase = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: str = 'counter_stark_amp',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            cal_parameter_name: Parameter('counter_stark_amp'),
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.
        }
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params, group=group)

        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            'counter_stark_amp',
            amplitudes_exp,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths,
            secondary_trajectory=secondary_trajectory,
            time_unit=time_unit
        )

        self.analysis = QutritCXCounterStarkAmpAnalysis(self.analysis.component_analysis())

        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, any]:
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
        group = self.experiment_options.group
        BaseUpdater.update(self._cals, experiment_data, self._param_name, self._sched_name,
                           group=group, fit_parameter=self._param_name)

        if self.experiment_options.update_cr_phase:
            raise NotImplementedError('')


class QutritCXCounterStarkAmpAnalysis(QutritCRHamiltonianScanAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.poly_orders = {(ic, ib): [0, 2] if ic == 0 and ib == 2 else 0
                               for ic in range(3) for ib in range(3)}
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Perform the fit on qutrit-basis operators.

        Additionally we compute the Hamiltonian components at zero Stark amp from the fit result.
        """
        analysis_results, figures = super()._run_additional_analysis(experiment_data, analysis_results, figures)

        omega_iz_coeffs = next(res.value for res in analysis_results if res.name == 'omega_Iz_coeffs')
        iz_order = PolynomialOrder(self.options.poly_orders[(0, 2)])

        stark_amp = 0.

        if iz_order.order == 1 and omega_iz_coeffs[1].n != 0.:
            stark_amp = -omega_iz_coeffs[0] / omega_iz_coeffs[1]
        elif iz_order.order == 2 and omega_iz_coeffs[2].n != 0.:
            det = omega_iz_coeffs[1] ** 2 - 4. * omega_iz_coeffs[0] * omega_iz_coeffs[2]
            if det.n < 0.:
                raise CalibrationError('Stark amplitude for omega_iz=0 not found')
            stark_amp = (-omega_iz_coeffs[1] + np.sign(omega_iz_coeffs[2].n) * unp.sqrt(det)) / (2. * omega_iz_coeffs[2])
        else:
            component_index = experiment_data.metadata["component_child_index"]

            control_0_data = experiment_data.child_data(component_index[0])
            amp_indices = control_0_data.metadata['component_child_index']
            amp_data = [control_0_data.child_data(idx) for idx in amp_indices]
            subanalysis = self._analyses[0]
            xvar = subanalysis.options.xvar
            xval = np.array([data.metadata[xvar] for data in amp_data])

            f = lambda x: unp.nominal_values(poly.polynomial.polyval(x, omega_iz_coeffs))
            # apparently providing fprime leads to instability because the coeff values are large
            sol = sciopt.root_scalar(f, x0=xval[0], x1=xval[-1])
            if not sol.converged:
                raise CalibrationError('Stark amplitude not found')

            stark_amp = ufloat(sol.root, 0.)

        analysis_results.append(AnalysisResultData(name='counter_stark_amp', value=stark_amp))

        h_components = np.empty((3, 3), dtype='O')

        for ic, control_op in enumerate(['I', 'z', 'ζ']):
            for ib, target_op in enumerate(['x', 'y', 'z']):
                coeffs = next(unp.nominal_values(res.value) for res in analysis_results
                              if res.name == f'omega_{control_op}{target_op}_coeffs')
                h_components[ic, ib] = poly.polynomial.polyval(stark_amp, coeffs)

        analysis_results.append(AnalysisResultData(name='hamiltonian_components', value=h_components))

        return analysis_results, figures


class QutritRCRCounterStarkAmpCal(BaseCalibrationExperiment, HamiltonianTomographyScan):
    """Determination of the counter Stark amplitude to cancel omega_2z via HT scan."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(0.01, 0.06, 6)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: str = 'counter_stark_amp',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            cal_parameter_name: Parameter(cal_parameter_name),
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.
        }
        schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                             assign_params=assign_params, group=group)

        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            cal_parameter_name,
            amplitudes_exp,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            rabi_init=cr_rabi_init(2),
            widths=widths,
            time_unit=time_unit
        )
        self.analysis.set_options(poly_orders=(0, 0, 2))

        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, any]:
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
        """Solve for the stark_amp where omega_z is 0."""
        group = self.experiment_options.group
        coeffs = unp.nominal_values(experiment_data.analysis_results('omega_z_coeffs', block=False).value)

        det = coeffs[1] ** 2 - 4. * coeffs[0] * coeffs[2]
        if det < 0.:
            warnings.warn('Stark amplitude for omega_2z=0 not found.', UserWarning)

        stark_amp = (-coeffs[1] + np.sign(coeffs[2]) * np.sqrt(det)) / (2. * coeffs[2])

        BaseUpdater.add_parameter_value(self._cals, experiment_data, stark_amp, self._param_name,
                                        schedule=self._sched_name, group=group)
