from typing import Iterable, Optional, Sequence, Tuple, Dict, Callable, List
import numpy as np
import numpy.polynomial as poly
from uncertainties import unumpy as unp
import scipy.optimize as sciopt

from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.framework import ExperimentData
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...common.util import PolynomialOrder
from ..qutrit_cr_hamiltonian import QutritCRHamiltonianScan

class QutritCRHamiltonianAmplitude02SyncCal(BaseCalibrationExperiment, QutritCRHamiltonianScan):
    def __init__(
        self,
        qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: List[str] = ['cr_amp', 'cr_phase'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Sequence[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            'cr_amp': Parameter('cr_amp'),
            'cr_width': Parameter('cr_width')
        }
        schedule = calibrations.get_schedule(
            schedule_name, qubits, assign_params=assign_params, group=group
        )

        if amplitudes is None:
            amplitudes = np.linspace(0.1, 0.8, 10)

        super().__init__(
            calibrations,
            qubits,
            schedule,
            'cr_amp',
            amplitudes,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            amplitudes=amplitudes,
            widths=widths,
            time_unit=time_unit
        )

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        metadata["cal_param_value"] = list(
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group,
            ) for pname in self._param_name
        )
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        group = self.experiment_options.group

        ## Find the cr_amp where omega_zx**2 + omega_zy**2 == omega_ζx**2 + omega_ζy**2
        omega_coeffs = list()
        for ic, control_op in enumerate(['z', 'ζ']):
            omega_coeffs.append(list())
            for ib, target_op in enumerate(['x', 'y']):
                result = experiment_data.analysis_results(f'omega_{control_op}{target_op}_coeffs', block=False).value
                omega_coeffs[-1].append(unp.nominal_values(result))

        # TODO take care of ragged case
        omega_coeffs = np.array(omega_coeffs)

        prime_coeffs = poly.polynomial.polyder(omega_coeffs)
        prime2_coeffs = poly.polynomial.polyder(omega_coeffs, m=2)

        omega_z = lambda amp: poly.polynomial.polyval(amp, omega_coeffs[:, 0])
        omega_zeta = lambda amp: poly.polynomial.polyval(amp, omega_coeffs[:, 1])
        omega_z_prime = lambda amp: poly.polynomial.polyval(amp, prime_coeffs[:, 0])
        omega_zeta_prime = lambda amp: poly.polynomial.polyval(amp, prime_coeffs[:, 1])
        omega_z_prime2 = lambda amp: poly.polynomial.polyval(amp, prime2_coeffs[:, 0])
        omega_zeta_prime2 = lambda amp: poly.polynomial.polyval(amp, prime2_coeffs[:, 1])

        def t_square(omega_c):
            return np.sum(np.square(omega_c), axis=0)

        def t_square_prime(omega_c, omega_c_prime):
            return np.sum(2. * omega_c_prime * omega_c, axis=0)

        def t_square_prime2(omega_c, omega_c_prime, omega_c_prime2):
            return np.sum(2. * (omega_c_prime2 * omega_c + np.square(omega_c_prime)), axis=0)

        f = lambda amp: t_square(omega_z(amp)) - t_square(omega_zeta(amp))
        fprime = lambda amp: (t_square_prime(omega_z(amp), omega_z_prime(amp))
                              - t_square_prime(omega_zeta(amp), omega_zeta_prime(amp)))
        fprime2 = lambda amp: (t_square_prime2(omega_z(amp), omega_z_prime(amp), omega_z_prime2(amp))
                               - t_square_prime2(omega_zeta(amp), omega_zeta_prime(amp), omega_zeta_prime2(amp)))

        amplitudes = self.experiment_options.amplitudes
        sol = sciopt.root_scalar(f, fprime=fprime, fprime2=fprime2, x0=amplitudes[0], x1=amplitudes[-1])
        cr_amp = sol.root

        ## Compute arctan(nu_zy / nu_zx) at the given cr_amp
        omega_z_val = omega_z(cr_amp)

        phase = np.arctan2(poly_z_val[1], poly_z_val[0])

        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, cr_amp, 'cr_amp', schedule=self._sched_name, group=group
        )
        BaseUpdater.add_parameter_value(
            self._cals, experiment_data, -phase, 'cr_phase', schedule=self._sched_name, group=group
        )
