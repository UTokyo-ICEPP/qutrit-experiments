from typing import Iterable, Optional, Dict, List, Sequence, Union, Tuple
import numpy as np
import numpy.polynomial as poly

from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Parameter
from qiskit.providers import Backend
from qiskit_experiments.framework import Options, ExperimentData
from qiskit_experiments.exceptions import CalibrationError
from qiskit_experiments.calibration_management import BaseCalibrationExperiment, Calibrations, ParameterValue
from qiskit_experiments.calibration_management.update_library import BaseUpdater

from ...calibrations.pulse_library import ModulatedGaussianSquare
from ...common.util import PolynomialOrder
from ..gs_amplitude import GSAmplitude

twopi = 2. * np.pi

class QutritCXCounterAmpCal(BaseCalibrationExperiment, GSAmplitude):
    """Set the amplitude of the counter pulse to induce the target IX and IY terms.

    The target ω should be a tuple (ωX, ωY) representing the Rabi frequencies that the counter tone
    should produce.
    """
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        target_omega: Tuple[float, float],
        schedule_name: str = 'cr',
        cal_parameter_name: List[str] = ['counter_amp', 'counter_phase'],
        auto_update: bool = True,
        group: str = 'default',
        backend: Optional[Backend] = None,
        amplitudes: Optional[Iterable[float]] = None,
        widths: Optional[Iterable[float]] = None,
        time_unit: Optional[float] = None,
        analysis_poly_order: PolynomialOrder = [1]
    ):
        assign_params = {
            'cr_amp': 0.,
            'cr_phase': 0.,
            'counter_amp': Parameter('amp'),
            'counter_stark_amp': 0.,
            'counter_phase': 0.,
            'cr_width': Parameter('width'),
            'cr_margin': 0.
        }
        cr_schedule = calibrations.get_schedule(schedule_name, physical_qubits,
                                                assign_params=assign_params, group=group)

        counter_pulse = next(block.pulse for block in cr_schedule.blocks
                             if block.name == 'Counter')

        with pulse.build(name='Counter') as schedule:
            pulse.play(counter_pulse, backend.drive_channel(physical_qubits[1]))

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

        for exp in self.component_experiment():
            exp.set_experiment_options(measured_logical_qubit=1)

        result_parameters = [f'freq_c{power}' for power in self.analysis.poly_order.powers]
        self.analysis.set_options(result_parameters=result_parameters)

        self.set_experiment_options(group=group)

        self.target_omega = target_omega

    def _metadata(self) -> Dict[str, any]:
        """Add metadata to the experiment data making it more self contained.
        The following keys are added to the experiment's metadata:
            cal_param_value: The value of the drive frequency parameter. This value together with
                the fit result will be used to find the new value of the drive frequency parameter.
            cal_param_name: The name of the parameter in the calibrations.
            cal_group: The calibration group to which the parameter belongs.
        """
        metadata = super()._metadata()
        try:
            metadata["cal_param_value"] = [
                self._cals.get_parameter_value(
                    pname,
                    self.physical_qubits,
                    schedule=self._sched_name,
                    group=self.experiment_options.group,
                ) for pname in self._param_name
            ]
        except CalibrationError:
            metadata["cal_param_value"] = None

        metadata["cal_target_omega"] = self.target_omega
        metadata["cal_group"] = self.experiment_options.group

        return metadata

    def _attach_calibrations(self, circuit: QuantumCircuit):
        pass

    def update_calibrations(self, experiment_data: ExperimentData):
        """Update the calibrations."""
        group = self.experiment_options.group
        index = self.experiment_options.result_index

        poly_order = self.analysis.poly_order
        freq_coeffs = np.zeros(poly_order.order + 1)
        for power in poly_order.powers:
            freq_coeffs[power] = BaseUpdater.get_value(experiment_data, f'freq_c{power}', index=index)

        target_abs_omega = np.sqrt(np.sum(np.square(self.target_omega)))
        target_phase = np.arctan2(*self.target_omega[::-1])

        if poly_order.order == 1:
            target_amp = target_abs_omega / freq_coeffs[1] / twopi
        else:
            raise NotImplementedError('Not seeing a need to think about this case at the moment')

        for value, pname in zip([target_amp, target_phase], self._param_name):
            BaseUpdater.add_parameter_value(self._cals, experiment_data, value, pname,
                                            self._sched_name, group)
