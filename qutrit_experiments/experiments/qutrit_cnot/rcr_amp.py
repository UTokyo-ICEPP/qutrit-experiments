from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import warnings
import numpy as np
import numpy.polynomial as poly
import scipy.linalg as scilin
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

from ...common.util import get_cr_margin, grounded_gauss_area
from ..qutrit_cr_hamiltonian import QutritCRHamiltonianScan, QutritCRHamiltonianScanAnalysis


twopi = 2. * np.pi

class QutritCXRCRAmpCal(BaseCalibrationExperiment, QutritCRHamiltonianScan):
    """Experiment to set up the basis for repeated CR.

    From a full three-block HT amplitude scan, we will determine
    - RCR type
    - Best CR amplitude
    - omega_Iz sign -> CR Stark frequency
    - rotary target omega_x
    - rough CR width
    - CX geometric phase
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.amplitudes = np.linspace(0.1, 0.8, 10)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: List[str] = ['rcr_type', 'cr', 'cr', 'cr', 'cr',
                                    'rotary_omega_x', 'cx_rabi', 'cx_rabi',
                                    'cx_geom_phase'],
        cal_parameter_name: List[str] = ['rcr_type', 'cr_width', 'cr_margin', 'stark_frequency', 'cr_amp',
                                         'rotary_omega_x', 'cx_rabi_offset', 'cx_rabi_frequency_diff',
                                         'cx_sign'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Sequence[float]] = None,
        widths: Optional[Iterable[float]] = None,
        secondary_trajectory: bool = False,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.,
            'cr_amp': Parameter('cr_amp'),
            'cr_stark_amp': 0.,
            'counter_amp': 0.,
            'counter_stark_amp': 0.
        }
        schedule = calibrations.get_schedule('cr', physical_qubits, assign_params=assign_params)

        if (amplitudes_exp := amplitudes) is None:
            amplitudes_exp = self._default_experiment_options().amplitudes

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            'cr_amp',
            amplitudes_exp,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths,
            secondary_trajectory=secondary_trajectory,
            time_unit=time_unit
        )

        if amplitudes is not None:
            self.set_experiment_options(amplitudes=amplitudes)

        self.analysis = QutritCXRCRAmpAnalysis(self.analysis.component_analysis())

        poly_orders = {
            (0, 0): 1,
            (0, 1): 1,
            (0, 2): 0,
            (1, 0): 3,
            (1, 1): 3,
            (1, 2): 2,
            (2, 0): 5,
            (2, 1): 5,
            (2, 2): 4
        }
        self.analysis.set_options(poly_orders=poly_orders)

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=sname,
                group=self.experiment_options.group,
            ) for pname, sname in zip(self._param_name, self._sched_name)
        ]

        cr_sigma = self._cals.get_parameter_value('cr_sigma', self._physical_qubits, 'cr')
        cr_rsr = self._cals.get_parameter_value('cr_rsr', self._physical_qubits, 'cr')
        x_duration = self._cals.get_parameter_value('duration', self._physical_qubits[0:1], 'x')
        x12_duration = self._cals.get_parameter_value('duration', self._physical_qubits[0:1], 'x12')
        metadata['risefall_duration_s'] = 2. * cr_rsr * cr_sigma * self._backend_data.dt
        metadata['risefall_area_s'] = (grounded_gauss_area(cr_sigma, cr_rsr, gs_factor=True)
                                       * self._backend_data.dt)
        metadata['x_duration_s'] = x_duration * self._backend_data.dt
        metadata['x12_duration_s'] = x12_duration * self._backend_data.dt

        properties = self.backend.properties()
        metadata['c_t2'] = properties.qubit_property(self._physical_qubits[0])['T2'][0]
        metadata['t_t2'] = properties.qubit_property(self._physical_qubits[1])['T2'][0]

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group
        result_index = self.experiment_options.result_index

        rcr_type = experiment_data.analysis_results('rcr_type', block=False).value

        # Calculate the width and margin in units of dt
        cr_width_s = BaseUpdater.get_value(experiment_data, 'cr_width_s', index=result_index)
        cr_width = cr_width_s / self._backend_data.dt
        cr_margin = get_cr_margin(cr_width, self.backend, self._cals, self.physical_qubits,
                                  self._sched_name[2])

        # Set a tentative Stark frequency depending on the sign of the Iz term
        omega_iz = BaseUpdater.get_value(experiment_data, 'omega_iz', index=result_index)
        drive_frequency = self._backend_data.drive_freqs[self.physical_qubits[1]]
        anharmonicity = self.backend.properties().qubit_property(self.physical_qubits[1],
                                                                 'anharmonicity')[0]
        if omega_iz > 0.:
            stark_frequency = drive_frequency + anharmonicity * 0.25
        else:
            stark_frequency = drive_frequency - anharmonicity * 0.25

        for pname, sname, value in zip(self._param_name[:4], self._sched_name[:4],
                                       [int(rcr_type), cr_width, cr_margin, stark_frequency]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=sname, group=group)

        for pname, sname in zip(self._param_name[4:], self._sched_name[4:]):
            BaseUpdater.update(self._cals, experiment_data, pname, sname,
                               group=group, fit_parameter=pname)


class QutritCXRCRAmpAnalysis(QutritCRHamiltonianScanAnalysis):
    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.drop_iz = True
        return options

    def _run_additional_analysis(
        self,
        experiment_data: ExperimentData,
        analysis_results: List[AnalysisResultData],
        figures: List["matplotlib.figure.Figure"]
    ) -> Tuple[List[AnalysisResultData], List["matplotlib.figure.Figure"]]:
        """Find the optimal CR amplitude and calculate other quantities."""
        # We get too many invalid value warnings
        err = np.seterr(invalid='ignore')

        analysis_results, figures = super()._run_additional_analysis(experiment_data,
                                                                     analysis_results, figures)

        np.seterr(**err)

        coeffs = {}
        for ic, cop in enumerate(['I', 'z', 'ζ']):
            for it, top in enumerate(['x', 'y', 'z']):
                coeffs[(ic, it)] = next(unp.nominal_values(res.value) for res in analysis_results
                                        if res.name == f'omega_{cop}{top}_coeffs')

        metadata = experiment_data.metadata

        component_index = metadata["component_child_index"]
        subanalysis = self._analyses[0]
        child_data = experiment_data.child_data(component_index[0])
        xvar = subanalysis.xvar
        xval = np.array([data.metadata[xvar] for data in child_data.child_data()])

        npoints = 100
        cr_amps = np.linspace(np.min(xval), np.max(xval), npoints)

        omegas_opbasis = np.empty((npoints, 3, 3))
        for (ic, it), coeff_arr in coeffs.items():
            omegas_opbasis[:, ic, it] = poly.polynomial.polyval(cr_amps, coeff_arr)

        omega_izs = omegas_opbasis[:, 0, 2].copy()
        if self.options.drop_iz:
            omegas_opbasis[:, 0, 2] = 0.

        # For each amplitude value, estimate the CRCR-based CX gate time and theoretical fidelity
        # for both RCR types. Find the amplitude with the highest fidelity * exp(-T2*gate_time)

        # Gate time determination:
        # - Approximate [U_CR]_j = exp(-i/2*(Ω_j*T + φ_j)x) (j: block index) where
        #   φ_j = ∫dt ω_j(t) and Ω_j is the flat-top Rabi frequency
        # - Approximate ω_j(t) = γ*a(t) for some constant and amplitude a(t) (the whole point
        #   of this experiment is that the relation is actually not linear, though). With the
        #   flat-top amplitude A, γ=Ω_j/A
        # - Then [U_CR]_j = exp(-i/2*Ω_j*(T + G/A)x) where G is the area of the pulse rise/fall
        # - U_RCR0 = exp(-i/2 * blkdiag[2*Ω_0, Ω_1+Ω_2, Ω_1+Ω_2] * (T + G/A) * x) and
        #   U_RCR2 = exp(-i/2 * blkdiag[Ω_0+Ω_1, Ω_0+Ω_1, 2*Ω_2] * (T + G/A) * x)
        # - U_CRCR0 = exp(-i/2 * blkdiag[2*Ω_0, 2*(Ω_1+Ω_2-Ω_0), 2*Ω_0] * (T + G/A) * x) and
        #   U_CRCR2 = exp(-i/2 * blkdiag[2*Ω_2, 2*(Ω_0+Ω_1-Ω_2), 2*Ω_2] * (T + G/A) * x)
        # - Rabi frequency differences are
        #   2*(Ω_1+Ω_2-2*Ω_0) = -6*Ω_z (CRCR0) and 2*(Ω_0+Ω_1-2*Ω_2) = 6*Ω_ζ (CRCR2)
        # - (T0 + G/A) * (-6*Ω_z) = -sgn(Ω_z)(π+2nπ), (T2 + G/A) * (6*Ω_ζ) = sgn(Ω_ζ)(π+2nπ) (n >= 0)
        #   The right hand sides are the Rabi phase differences in the final CX.
        # - T0 = (π+2nπ) / |6*Ω_z| - G/A, T2 = (π+2nπ) / |6*Ω_ζ| - G/A
        # - For proto_CX = blkdiag(I, Sj * iX, I),
        #   S0 = sgn(Ω_z)*(-1)^n, S2 = -sgn(Ω_ζ)*(-1)^n

        # Normalized gaussian area = G/A
        risefall_area = metadata['risefall_area_s']

        rcr_types = ['0', '2']

        paulis = np.array([
            [[0., 1.], [1., 0.]],
            [[0., -1.j], [1.j, 0.]],
            [[1., 0.], [0., -1.]]
        ])

        # Hamiltonian components with rotary tone
        omegas = {}
        rotary_omega_xs = {}
        for rcr_type, i, j in [('0', 1, 2), ('2', 0, 1)]:
            om = np.einsum('co,aot->act',
                           np.array([[1, 1, 0], [1, -1, 1], [1, 0, -1]]),
                           omegas_opbasis)
            # Rotary tones:
            #  (ω_ix+Δωx)*ω_jz - (ω_jx+Δωx)*ω_iz = 0
            #  => Δωx = (ω_ix*ω_jz - ω_jx*ω_iz) / (ω_iz - ω_jz)
            numer = om[:, i, 0] * om[:, j, 2] - om[:, j, 0] * om[:, i, 2]
            denom = om[:, i, 2] - om[:, j, 2]
            rotary_omega_xs[rcr_type] = np.divide(numer, denom, out=np.zeros_like(cr_amps),
                                                  where=np.logical_not(np.isclose(denom, 0.)))
            om[:, :, 0] += rotary_omega_xs[rcr_type][:, None]
            omegas[rcr_type] = om

        # Calculate T0, T2, and sign of block 1 in protoCX=blkdiag(I, ±iX, I)
        cr_widths = {}
        cx_signs = {}
        cx_rabi_frequency_diffs = {}
        for rcr_type, sign, basis in zip(rcr_types, [1., -1.], [(1, 0), (2, 0)]): # zx and ζx
            omega = omegas_opbasis[(slice(None),) + basis]
            denom = 6. * np.abs(omega)
            abs_pi_over_sixomega = np.ones_like(denom) # if denom is zero, CR width is set to 1 sec
            np.divide(np.pi, denom, out=abs_pi_over_sixomega, where=(denom != 0.))
            w = abs_pi_over_sixomega - risefall_area
            s = sign * np.sign(omega, out=np.ones_like(omega), where=(omega != 0.))
            while np.any(w < 0.):
                s *= np.where(w < 0., -1., 1.)
                w += np.where(w < 0., 2. * abs_pi_over_sixomega, 0.)

            cr_widths[rcr_type] = w
            cx_signs[rcr_type] = s
            cx_rabi_frequency_diffs[rcr_type] = -sign * 6. * omega / twopi

        # Compute the theoretical fidelity
        pulse_areas = {rcr_type: w + risefall_area for rcr_type, w in cr_widths.items()}
        h_cr = {rcr_type: np.einsum('a,act,tij->acij',
                                    pulse_areas[rcr_type], comps, paulis)
                for rcr_type, comps in omegas.items()}
        u_cr = {}
        for rcr_type, h in h_cr.items():
            u_cr[rcr_type] = {sign: scilin.expm(-0.5j * h) for sign in ['+', '-']}
            # Inverted pulse phase -> inverted sign of unitary off-diagonals
            u_cr[rcr_type]['-'][:, :, [0, 1], [1, 0]] *= -1.

        u_rcr = {'0' + sign: u[:, [0, 2, 1]] @ u for sign, u in u_cr['0'].items()}
        u_rcr |= {'2' + sign: u[:, [1, 0, 2]] @ u for sign, u in u_cr['2'].items()}
        u_crcr = {'0': u_rcr['0-'][:, [2, 0, 1]] @ u_rcr['0+'][:, [1, 2, 0]]
                       @ u_rcr['0+'],
                  '2': u_rcr['2+'][:, [2, 0, 1]] @ u_rcr['2-'][:, [1, 2, 0]]
                       @ u_rcr['2+']}
        h_rx = {rcr_type: -2. * np.einsum('a,ij->aij',
                                          omegas[rcr_type][:, int(rcr_type), 0] * area,
                                          paulis[0])
                for rcr_type, area in pulse_areas.items()}
        u_rx = {rcr_type: scilin.expm(-0.5j * exponent)
                for rcr_type, exponent in h_rx.items()}

        u_crcr_rx = {rcr_type: np.expand_dims(u_rx[rcr_type], 1) @ u
                     for rcr_type, u in u_crcr.items()}

        # diff / pi -> .., -3, -1, 1, 3, ..
        # CX sign   -> .., -i, i, -i, i, ..
        ideal_cx = {rcr_type: np.stack([
            np.tile([[[1., 0.], [0., 1.]]], (npoints, 1, 1)),
            1.j * sign[:, None, None] * paulis[0],
            np.tile([[[1., 0.], [0., 1.]]], (npoints, 1, 1))
        ], axis=1) for rcr_type, sign in cx_signs.items()}
        fidelity = {rcr_type: np.square(np.abs(
                                  np.einsum('acij,acij->a', u, np.conjugate(ideal_cx[rcr_type]))
                              )) / 36.
                    for rcr_type, u in u_crcr_rx.items()}

        # Compute the gate time
        gate_duration = {rcr_type: 3. * (width + metadata['risefall_duration_s']
                                         + metadata['x_duration_s'] + metadata['x12_duration_s'])
                         for rcr_type, width in cr_widths.items()}
        decoherence_factor = {rcr_type: np.exp(-d / metadata['c_t2']) * np.exp(-d / metadata['t_t2'])
                              for rcr_type, d in gate_duration.items()}

        # Find the best RCR type and amplitude
        objective = {rcr_type: np.amax(fidelity[rcr_type] * df)
                     for rcr_type, df in decoherence_factor.items()}
        rcr_type = max(rcr_types, key=lambda t: objective[t])
        imax = np.argmax(fidelity[rcr_type] * decoherence_factor[rcr_type])

        cx_rabi_offset = (2. * omegas[rcr_type][imax, int(rcr_type), 0]
                          * (cr_widths[rcr_type][imax] + risefall_area))

        analysis_results += [
            AnalysisResultData(name='rcr_type', value=rcr_type),
            AnalysisResultData(name='cr_amp', value=ufloat(cr_amps[imax], 0.)),
            AnalysisResultData(name='cr_width_s', value=ufloat(cr_widths[rcr_type][imax], 0.)),
            AnalysisResultData(name='rotary_omega_x',
                               value=ufloat(rotary_omega_xs[rcr_type][imax], 0.)),
            AnalysisResultData(name='omega_iz', value=ufloat(omega_izs[imax], 0.)),
            AnalysisResultData(name='cx_rabi_offset', value=ufloat(cx_rabi_offset, 0.)),
            AnalysisResultData(name='cx_rabi_frequency_diff',
                               value=ufloat(cx_rabi_frequency_diffs[rcr_type][imax], 0.)),
            AnalysisResultData(name='cx_sign', value=ufloat(cx_signs[rcr_type][imax], 0.)),
            AnalysisResultData(name='fidelity', value=fidelity),
            AnalysisResultData(name='decoherence_factor', value=decoherence_factor)
        ]

        return analysis_results, figures
