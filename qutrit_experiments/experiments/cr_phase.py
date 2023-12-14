from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.framework import ExperimentData, AnalysisResultData, Options
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from .cr_rabi import cr_rabi_init
from .hamiltonian_tomography import HamiltonianTomography, HamiltonianTomographyAnalysis


twopi = 2. * np.pi

class CRPhase(HamiltonianTomography):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        backend: Optional[Backend] = None
    ):
        if backend is None:
            raise RuntimeError('CRPhaseCal requires a backend')

        # Should we use 0X or 1X? To first order, ωIx±ωzx = -JΩ/(Δct+αc)*(1±αc/Δct)
        # -> Check |1±αc/Δct| and take the sign with a greater value
        c_props = backend.target.qubit_properties[physical_qubits[0]]
        t_props = backend.target.qubit_properties[physical_qubits[1]]
        alpha_c = c_props.anharmonicity
        delta_ct = c_props.frequency - t_props.frequency

        if abs(1. + alpha_c / delta_ct) > abs(1. - alpha_c / delta_ct):
            # Measure 0X
            control_state = 0
            expected_sign = -1. * np.sign(delta_ct)
        else:
            control_state = 1
            expected_sign = -1. * np.sign((1. - alpha_c / delta_ct) / (delta_ct + alpha_c))

        super().__init__(physical_qubits, schedule, rabi_init=cr_rabi_init(control_state),
                         widths=widths, time_unit=time_unit, backend=backend)

        self.extra_metadata['expected_sign'] = expected_sign


class CRPhaseCal(BaseCalibrationExperiment, CRPhase):
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        schedule_name: str = 'cr',
        cal_parameter_name: str = 'cr_phase',
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None
    ):
        assign_params = {
            'cr_amp': 0.4,
            cal_parameter_name: 0.,
            'cr_stark_amp': 0.,
            'cr_width': Parameter('cr_width'),
            'cr_margin': 0.,
            'cr_stark_amp': 0.,
            'counter_amp': 0.
        }
        schedule = calibrations.get_schedule(
            schedule_name, physical_qubits, assign_params=assign_params, group=group
        )

        super().__init__(
            calibrations,
            physical_qubits,
            schedule,
            schedule_name=schedule_name,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            backend=backend,
            widths=widths,
            time_unit=time_unit
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

        h_components = experiment_data.analysis_results('hamiltonian_components', block=False).value
        h_components = unp.nominal_values(h_components)
        observed_angle = np.arctan2(h_components[1], h_components[0])

        # In circuit notation (left to right):
        #   Pulse(angle=φ) = ShiftPhase(φ) Pulse ShiftPhase(-φ)
        # On IBM backends, ShiftPhase(φ) effects a +φ Z rotation of the state vector on the Bloch
        # sphere (contrary to how rz(φ) is scheduled). Therefore the axis of the polar rotation
        # by Pulse(angle=φ) is rotated -φ from the original Pulse.
        # Thus
        #   observed_angle = -current_phase + hardware_offset
        #   target_angle = -new_phase + hardware_offset
        # but current_phase is always 0 (set in __init__)
        #   -> new_phase = observed_angle - target_angle
        # tone to point towards target_angle when Pulse(angle=current_phase) points it to
        # omega_0_angle.

        target_angle = 0. if experiment_data.metadata['expected_sign'] > 0. else np.pi

        new_phase = observed_angle - target_angle
        if new_phase > np.pi:
            new_phase -= twopi
        elif new_phase < -np.pi:
            new_phase += twopi

        BaseUpdater.add_parameter_value(self._cals, experiment_data, new_phase, self._param_name,
                                        schedule=self._sched_name, group=group)
