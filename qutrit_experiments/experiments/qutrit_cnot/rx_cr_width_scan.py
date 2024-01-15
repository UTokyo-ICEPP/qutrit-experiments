from typing import Sequence, Optional, List, Dict, Any, Sequence, Tuple
import numpy as np
from uncertainties import ufloat, unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import CXGate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit.result import Counts
from qiskit_experiments.framework import (BaseExperiment, Options, ExperimentData, AnalysisResultData,
                                          BackendData, BackendTiming)
from qiskit_experiments.calibration_management import (Calibrations, ParameterValue,
                                                       BaseCalibrationExperiment)
from qiskit_experiments.calibration_management.update_library import BaseUpdater
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.curve_analysis.base_curve_analysis import PARAMS_ENTRY_PREFIX
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ...common.transpilation import replace_calibration_and_metadata
from ...common.gates import SetF12
from ...common.util import default_shots, get_cr_margin
from ...common.framework_overrides import BatchExperiment, CompoundAnalysis
from ...common.linked_curve_analysis import LinkedCurveAnalysis
from ..fine_amplitude import CustomTranspiledFineAmplitude
from ..cr_rabi import CRRabi
from ..dummy_data import single_qubit_counts

twopi = 2. * np.pi


class QutritCXRxScan(BaseExperiment):
    r"""Rabi oscillation experiment varying the amplitude of an Rx pulse in the CX seqeuence.

    CX unitary has a form

    .. math::

        G_{\mathrm{CX}} = \mathrm{blkdiag} \begin{pmatrix}
            e^{-\frac{i}{2} [\nu_{0x} t + \phi_0 + \rho(A)] x} \\
            e^{-\frac{i}{2} [\nu_{1x} t + \phi_1 + \rho(A)] x} \\
            e^{-\frac{i}{2} [\nu_{2x} t + \phi_2 + \rho(A)] x}
        \end{pmatrix}

    where nominally

    .. math::
        \nu_{0x} = -2\omega_{zx} + \omega_{Ix} \\
        \nu_{1x} = 2\omega_{zx} - 2\omega_{\zeta x} + \omega_{Ix} \\
        \nu_{2x} = 2\omega_{\zeta x} + \omega_{Ix}

    using the CR effective Hamiltonian components :math:`\omega_{gx}`, and :math:`\rho` is the Rabi
    phase offset from the Rx pulse with amp :math:`A` in the schedule. Since by this point the CR
    and counter tones have been calibrated to have :math:`\nu_{0x} = \nu_{2x} = 0` and
    :math:`\nu_{1x} = 6\omega_{zx}`, what we are after are the CR pulse width :math:`t_{\mathrm{CX}}`
    and :math:`A_{\mathrm{CX}}` that results in

    .. math::

        \phi_0 + \rho(A_{\mathrm{CX}}) = 0
        \nu_{1x} t_{\mathrm{CX}} + \phi_1 - \phi_0 = \pm \pi.

    We can determine :math:`A_{\mathrm{CX}}` and :math:`t_{\mathrm{CX}}` individually (originally
    we had designed a convoluted experiment determining both simultaneously). This experiment is for
    finding the former.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.control_state = None
        options.experiment_index = None
        options.rx_amps = np.linspace(-0.2, 0.2, 33)
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        control_state: int,
        backend: Optional[Backend] = None,
        rx_amps: Optional[Sequence[float]] = None,
        experiment_index: Optional[int] = None
    ):
        super().__init__(physical_qubits, backend=backend, analysis=curve.OscillationAnalysis())

        if control_state == 0:
            phase_p0 = 0.
        else:
            phase_p0 = np.pi

        self.analysis.set_options(
            outcome='0',
            result_parameters=['freq', 'phase'],
            normalization=False,
            p0={'amp': 0.5, 'base': 0.5, 'phase': phase_p0},
            bounds={'amp': (0., 1.), 'base': (0., 1.), 'freq': (0., np.inf), 'phase': (-np.pi, np.pi)}
        )
        self.analysis.plotter.set_figure_options(
            ylim=(-0.1, 1.1),
            xlabel='Rx amplitude',
            ylabel='P(0)'
        )

        if backend:
            x_sched = backend.defaults().instruction_schedule_map.get('x', physical_qubits[1])
            x_pulse = next(inst.pulse for _, inst in x_sched.instructions if isinstance(inst, pulse.Play))
            self.analysis.options.p0['freq'] = 0.5 / np.abs(x_pulse.amp)

        self.set_experiment_options(schedule=schedule, control_state=control_state)

        if rx_amps is not None:
            self.set_experiment_options(rx_amps=rx_amps)
        # Needed for LinkedCurveAnalysis
        if experiment_index is not None:
            self.set_experiment_options(experiment_index=experiment_index)

    def circuits(self) -> List[QuantumCircuit]:
        schedule = self.experiment_options.schedule

        rx_amp = schedule.get_parameters('rx_amp')[0]

        gate = Gate(schedule.name, 2, [])

        template = QuantumCircuit(2, 1)
        if self.experiment_options.control_state == 1:
            template.x(0)
        template.append(gate, [0, 1])
        template.measure(1, 0)

        template.metadata = {'qubits': self.physical_qubits}
        if self.experiment_options.experiment_index is not None:
            template.metadata['experiment_index'] = self.experiment_options.experiment_index

        circuits = []
        for amp_val in self.experiment_options.rx_amps:
            sched = schedule.assign_parameters({rx_amp: amp_val}, inplace=False)

            circuit = template.copy()
            circuit.add_calibration(gate, self.physical_qubits, sched, [])
            circuit.metadata['xval'] = amp_val
            circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        return replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                self.transpile_options.target)

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()

        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        return metadata

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[Counts]:
        shots = self.run_options.get('shots', default_shots)
        rx_amps = np.array(self.experiment_options.rx_amps)
        probs = 0.5 + 0.5 * np.cos(np.pi / 0.14 * rx_amps + 0.12)
        return single_qubit_counts(probs, shots)


class QutritCXRxCal(BaseCalibrationExperiment, QutritCXRxScan):
    r"""Rx amplitude determination.

    - Case 1. Direct CR: Assume :math:`\nu_{0x}=0`. `\phi_0` is the residual phase coming from the
      nonlinearity of the oscillation during rise and fall. Set the width to 0 to isolate `\phi_0`.
    - Case 2. Cycled repeated CR: :math:`U_{\mathrm{CX}} = \mathrm{blkdiag}(Rx(\phi_0),
      Rx(\phi_0 \pm \pi), Rx(\phi_0))`. Simply run the default CX sequence and identify :math:`\phi_0`.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = ['rx', 'rx', 'cx_rabi'],
        cal_parameter_name: str = ['rx_amp', 'rx_angle', 'rx_rabi_frequency'],
        auto_update: bool = True,
        group: str = 'default',
        zero_width: bool = True,
        backend: Optional[Backend] = None,
        rx_amps: Optional[Sequence[float]] = None
    ):
        physical_qubits = tuple(physical_qubits)

        set_f12 = calibrations.get_schedule('set_f12', physical_qubits[0], group=group)
        assign_params = {
            (cal_parameter_name[0], physical_qubits, schedule_name[0]): Parameter(cal_parameter_name[0]),
            (cal_parameter_name[1], physical_qubits, schedule_name[1]): 0.
        }
        if zero_width:
            assign_params.update({
                ('cr_width', physical_qubits, 'cr'): 0.,
                ('cr_margin', physical_qubits, 'cr'): 0.
            })
        qutrit_cx = calibrations.get_schedule(
            'qutrit_cx', physical_qubits, assign_params=assign_params, group=group
        )

        with pulse.build(name='cx') as schedule:
            pulse.call(set_f12)
            pulse.call(qutrit_cx)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            0,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            rx_amps=rx_amps
        )

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()
        metadata["cal_param_value"] = [
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=sname,
                group=self.experiment_options.group
            ) for pname, sname in zip(self._param_name, self._sched_name)
        ]
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        phi0 = BaseUpdater.get_value(experiment_data, 'phase', index=result_index)
        rabi_freq = BaseUpdater.get_value(experiment_data, 'freq', index=result_index)
        rx_amp = -phi0 / rabi_freq / twopi
        rx_angle = 0.
        if rx_amp < 0.:
            rx_amp = abs(rx_amp)
            rx_angle = np.pi

        for pname, sname, value in zip(self._param_name, self._sched_name, [rx_amp, rx_angle, rabi_freq]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=sname, group=group)


class QutritCXRoughCRWidthCal(BaseCalibrationExperiment, CRRabi):
    r"""CR width determination assuming CCR sequence unitary of form blkdiag[I, Rx, I]."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: List[str] = ['cr_width', 'cr_margin'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        widths: Optional[np.ndarray] = None
    ):
        physical_qubits = tuple(physical_qubits)

        set_f12 = calibrations.get_schedule('set_f12', physical_qubits[0], group=group)
        assign_params = {
            ('cr_width', physical_qubits, 'cr'): Parameter('cr_width'),
            ('cr_margin', physical_qubits, 'cr'): 0.
        }
        qutrit_cx = calibrations.get_schedule('qutrit_cx', physical_qubits,
                                              assign_params=assign_params, group=group)

        with pulse.build(name='cx') as schedule:
            pulse.call(set_f12)
            pulse.call(qutrit_cx)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            1,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths
        )

        self.analysis.set_options(
            result_parameters=['freq', 'phase'],
            normalization=False
        )

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
        metadata["cal_param_value"] = list(
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=self._sched_name,
                group=self.experiment_options.group,
            ) for pname in self._param_name
        )
        metadata["cal_group"] = self.experiment_options.group

        cr_sched = self._cals.get_schedule('cr', self.physical_qubits)
        cr_pulse = next(inst.pulse for t, inst in cr_sched.instructions if inst.name == 'CR')
        metadata['risefall'] = cr_pulse.support - cr_pulse.width

        if self.backend:
            metadata['granularity'] = BackendData(self.backend).granularity
        else:
            metadata['granularity'] = 1

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group
        result_index = self.experiment_options.result_index

        offset = BaseUpdater.get_value(experiment_data, 'phase', index=result_index)
        freq = BaseUpdater.get_value(experiment_data, 'freq', index=result_index)

        cr_width = (np.pi - offset) / freq / twopi
        while cr_width > 0.:
            cr_width -= 1. / np.abs(freq)
        while cr_width < 0.:
            cr_width += 1. / np.abs(freq)
        cr_width /= self._backend_data.dt

        cr_margin = get_cr_margin(cr_width, self.backend, self._cals, self.physical_qubits,
                                  self._sched_name[0])

        for pname, value in zip(self._param_name, [cr_width, cr_margin]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=self._sched_name, group=group)


class CXPingPong(CustomTranspiledFineAmplitude):
    r"""CR width determination for a CR sequence that roughly produces blkdiag(I, x, I)."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        control_state: int = 1,
        schedule: Optional[ScheduleBlock] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, CXGate(), backend=backend,
                         measurement_qubits=[physical_qubits[1]])
        self.analysis.set_options(
            outcome='1',
            fixed_parameters={
                'angle_per_gate': np.pi if control_state == 1 else twopi,
                'phase_offset': np.pi / 2.
            }
        )
        self.control_state = control_state
        self.schedule = schedule

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        circuit = QuantumCircuit(self.num_qubits, num_clbits)
        if self.control_state == 1:
            circuit.x(0)
        circuit.sx(1)
        return circuit

    def circuits(self) -> List[QuantumCircuit]:
        circuits = super().circuits()
        for circuit in circuits:
            # Necessary for readout mitigation
            if self.schedule is not None:
                circuit.add_calibration('cx', self.physical_qubits, self.schedule)

        return circuits

    def dummy_data(self, transpiled_circuits: List[QuantumCircuit]) -> List[Counts]:
        repetitions = np.array(self.experiment_options.repetitions)
        shots = self.run_options.get('shots', default_shots)
        probs = 0.5 * np.cos(np.pi / 2. + 0.02 + (np.pi + 0.01) * repetitions) + 0.5
        probs = np.concatenate(([1., 0.], probs))

        return single_qubit_counts(probs, shots)


class QutritCXFineCRWidthCal(BaseCalibrationExperiment, CXPingPong):
    r"""CR width determination for :math:`A_{\mathrm{CX}}` assuming :math:`\nu_{0x}=0` and
    :math:`\nu_{1x}=6\omega_{zx}`."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        omega_sign: float,
        rabi_phase_offset: float,
        schedule_name: str = 'cr',
        cal_parameter_name: List[str] = ['cr_width', 'cr_margin'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None
    ):
        super().__init__(
            calibrations,
            physical_qubits,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend
        )

        set_f12 = calibrations.get_schedule('set_f12', physical_qubits[0], group=group)
        qutrit_cx = calibrations.get_schedule('qutrit_cx', physical_qubits, group=group)
        phase_offset = 0. if omega_sign > 0. else np.pi

        with pulse.build(name='cx') as cx_sched:
            pulse.call(set_f12)
            with pulse.phase_offset(phase_offset, *qutrit_cx.channels):
                pulse.call(qutrit_cx)

        self.cx_sched = cx_sched
        self.rabi_phase_offset = rabi_phase_offset

    def _attach_calibrations(self, circuit: QuantumCircuit):
        circuit.add_calibration('cx', self.physical_qubits, self.cx_sched, [])

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

        cr_pulse = next(inst.pulse for t, inst in self.cx_sched.instructions if inst.name == 'CR')
        metadata['risefall'] = cr_pulse.support - cr_pulse.width

        if self.backend:
            metadata['granularity'] = BackendData(self.backend).granularity
        else:
            metadata['granularity'] = 1

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        group = self.experiment_options.group

        current_width = experiment_data.metadata['cal_param_value'][0]
        d_theta = BaseUpdater.get_value(experiment_data, 'd_theta')
        target_angle = np.pi - self.rabi_phase_offset

        cr_width = current_width * target_angle / (target_angle + d_theta)
        risefall = experiment_data.metadata['risefall']
        support = cr_width + risefall
        granularity = experiment_data.metadata['granularity']
        cr_duration = np.ceil(support / granularity) * granularity
        cr_margin = cr_duration - support

        for pname, value in zip(self._param_name, [cr_width, cr_margin]):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            schedule=self._sched_name, group=group)


class QutritCXCRWidthScan(BatchExperiment):
    r"""Experiment to determine :math:`t_{\mathrm{CX}}` without assuming :math:`\nu_{0x}=0`.

    Idea:
    1. Fix a CR width and scan the Rx amp.
       -> Perform an oscillation analysis wrt Rx amp. Determine the offsetting amp from phase and freq.
    2. Scan the CR width.
       -> Perform a LinkedCurveAnalysis of the oscillation analyses with a common freq and phase linear on width.
       -> Find the offset amplitude as a function of the CR width.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        control_state: int,
        cr_widths: Sequence[float],
        backend: Optional[Backend] = None,
        rx_amps: Optional[Sequence[float]] = None
    ):
        cr_width = schedule.get_parameters('cr_width')[0]
        cr_margin = schedule.get_parameters('cr_margin')[0]

        if backend:
            granularity = BackendData(backend).granularity
        else:
            granularity = 1

        test_sched = schedule.assign_parameters({cr_width: 0., cr_margin: 0.}, inplace=False)
        cr_pulse = next(inst.pulse for t, inst in test_sched.instructions if inst.name == 'CR')
        risefall = cr_pulse.duration

        experiments = []
        analyses = []

        for idx, width_val in enumerate(cr_widths):
            support = width_val + risefall
            duration = np.ceil(support / granularity) * granularity
            margin_val = duration - support

            sched = schedule.assign_parameters({cr_width: width_val, cr_margin: margin_val}, inplace=False)
            exp = QutritCXRxScan(physical_qubits, sched, control_state, backend=backend,
                                 rx_amps=rx_amps, experiment_index=idx)
            exp.set_experiment_options(cr_width=width_val)
            experiments.append(exp)
            analyses.append(exp.analysis)

        phase_func = 'polynomial.polynomial.polyval(cr_width, [phase_c0, phase_c1])'

        analysis = LinkedCurveAnalysis(
            analyses,
            linked_params={
                'amp': None,
                'base': None,
                'freq': None,
                'phase': phase_func
            },
            experiment_params=['cr_width']
        )

        analysis.set_options(
            p0={
                'phase_c0': np.pi if control_state == 1 else 0.,
                'phase_c1': 0.
            }
        )

        super().__init__(experiments, analysis=analysis)


class QutritCXCRWidthScanOffsetDiff(BatchExperiment):
    r"""Experiment to determine :math:`t_{\mathrm{CX}}` without assuming :math:`\nu_{0x}=0`.

    Idea:
    1. Fix a CR width and scan the Rx amp.
       -> Perform an oscillation analysis wrt Rx amp. Determine the offsetting amp from phase and freq.
    2. Scan the CR width.
       -> Perform a LinkedCurveAnalysis of the oscillation analyses with a common freq and phase linear on width.
       -> Find the Rabi phase at Rx amp 0 as a function of the CR width.
    3. Repeat the CR width scan for control state 0 and 1.
       -> Rabi phase difference between 0 and 1 does not depend on the Rx amplitude. Find the width which makes the
          interblock difference π
       -> Also find the Rx amp that cancels the offset in block 0.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        cr_widths: Sequence[float],
        backend: Optional[Backend] = None,
        rx_amps: Optional[Sequence[float]] = None
    ):
        experiments = []
        analyses = []
        for control_state in range(2):
            exp = QutritCXCRWidthScan(physical_qubits, schedule, control_state, cr_widths,
                                      backend=backend, rx_amps=rx_amps)
            experiments.append(exp)
            analyses.append(exp.analysis)

        super().__init__(experiments, analysis=QutritCXCRWidthScanOffsetDiffAnalysis(analyses))


class QutritCXCRWidthScanOffsetDiffAnalysis(CompoundAnalysis):
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
        component_index = experiment_data.metadata["component_child_index"]
        child_data = [experiment_data.child_data(idx) for idx in component_index]

        phase_coeffs = []
        freqs = []
        for control_state in range(2):
            subanalysis = self._analyses[control_state]
            res_name = PARAMS_ENTRY_PREFIX + subanalysis.name
            popt = child_data[control_state].analysis_results(res_name).value.ufloat_params
            phase_coeffs.append(np.array([popt['phase_c0'], popt['phase_c1']]))
            freqs.append(popt['freq'])

        offset_diff_coeffs = phase_coeffs[1] - phase_coeffs[0]

        # Shouldn't there be a dt here?
        # Also check the sign
        cr_width = (np.pi - offset_diff_coeffs[0]) / offset_diff_coeffs[1]
        phase_0 = phase_coeffs[0][0] + phase_coeffs[0][1] * cr_width
        # twopi * freqs[0] * rx_amp + phase_0 = 0
        rx_amp = -phase_0 / twopi / freqs[0]

        analysis_results.extend([
            AnalysisResultData(name='cr_width', value=cr_width),
            AnalysisResultData(name='rx_amp', value=rx_amp)
        ])

        if self.options.plot:
            scan_component_index = child_data[0].metadata['component_child_index']
            imin = scan_component_index[0]
            imax = scan_component_index[-1]

            xmin = child_data[0].child_data(imin).metadata['cr_width']
            xmax = child_data[0].child_data(imax).metadata['cr_width']
            interp_x = np.linspace(xmin, xmax, 100)

            y_data = offset_diff_coeffs[0] + offset_diff_coeffs[1] * interp_x

            plotter = CurvePlotter(MplDrawer())

            plotter.set_figure_options(
                xlabel='CR width',
                ylabel='phase offset diff'
            )

            plotter.set_series_data(
                'offset_diff',
                x_interp=interp_x,
                y_interp=unp.nominal_values(y_data),
            )

            plotter.set_series_data(
                'offset_diff',
                y_interp_err=unp.std_devs(y_data)
            )

            figures.append(plotter.figure())

        return analysis_results, figures


class QutritCXCRWidthScanOffsetDiffCal(BaseCalibrationExperiment, QutritCXCRWidthScanOffsetDiff):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: List[str] = ['cr', 'cr', 'qutrit_cx'],
        cal_parameter_name: List[str] = ['cr_width', 'cr_margin', 'rx_amp'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        rx_amps: Optional[Sequence[float]] = None
    ):
        set_f12 = calibrations.get_schedule('set_f12', physical_qubits[0], group=group)

        rough_cr_width = calibrations.get_parameter_value(cal_parameter_name[0],
                                                          physical_qubits,
                                                          schedule=schedule_name[0],
                                                          group=group)
        cr_widths = np.linspace(max(rough_cr_width - 16., 0), rough_cr_width + 16., 5)

        assign_params = {(pname, tuple(physical_qubits), sname): Parameter(pname)
                         for sname, pname in zip(schedule_name, cal_parameter_name)}
        qutrit_cx = calibrations.get_schedule(
            'qutrit_cx', physical_qubits, assign_params=assign_params, group=group
        )

        with pulse.build(name='cx') as schedule:
            pulse.call(set_f12)
            pulse.call(qutrit_cx)

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            cr_widths,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            rx_amps=rx_amps
        )

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
        metadata["cal_param_value"] = list(
            self._cals.get_parameter_value(
                pname,
                self.physical_qubits,
                schedule=sname,
                group=self.experiment_options.group,
            ) for pname, sname in zip(self._param_name, self._sched_name)
        )
        metadata["cal_group"] = self.experiment_options.group

        assign_params = {(pname, self.physical_qubits, sname): 0.
                         for pname, sname in zip(self._param_name[:2], self._sched_name[:2])}
        test_sched = self._cals.get_schedule(
            'qutrit_cx', self.physical_qubits,
            assign_params=assign_params,
            group=self.experiment_options.group
        )
        cr_pulse = next(inst.pulse for t, inst in test_sched.instructions if inst.name == 'CR')
        metadata['risefall'] = cr_pulse.duration

        if self.backend:
            metadata['granularity'] = BackendData(self.backend).granularity
        else:
            metadata['granularity'] = 1

        return metadata

    def update_calibrations(self, experiment_data: ExperimentData):
        result_index = self.experiment_options.result_index
        group = self.experiment_options.group

        for pname, sname in zip(self._param_name[0::2], self._sched_name[0::2]):
            BaseUpdater.update(self._cals, experiment_data, pname,
                               sname, result_index=result_index,
                               group=group, fit_parameter=pname)

        cr_width = BaseUpdater.get_value(experiment_data, 'cr_width')
        risefall = experiment_data.metadata['risefall']
        support = cr_width + risefall
        granularity = experiment_data.metadata['granularity']
        cr_duration = np.ceil(support / granularity) * granularity
        cr_margin = cr_duration - support

        BaseUpdater.add_parameter_value(self._cals, experiment_data, cr_margin, self._param_name[1],
                                        schedule=self._sched_name[1], group=group)
