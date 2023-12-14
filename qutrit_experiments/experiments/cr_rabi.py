from typing import Iterable, Optional, Sequence
from functools import partial
import numpy as np

from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock

from qiskit_experiments.framework import Options
from qiskit_experiments.exceptions import CalibrationError

from .gs_rabi import GSRabi
from ..common.gates import SetF12, X12Gate

class CRRabi(GSRabi):
    """Experiment to observe Rabi oscillation of the target qubit by a CR tone on the control qubit."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()

        options.control_state = None
        options.measured_logical_qubit = 1
        options.widths = np.linspace(0., 2048., 17)
        options.param_name = 'cr_width'

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        control_state: int,
        widths: Optional[Iterable[float]] = None,
        initial_state: Optional[str] = None,
        meas_basis: Optional[str] = None,
        time_unit: Optional[float] = None,
        experiment_index: Optional[int] = None,
        backend: Optional[Backend] = None
    ):
        super().__init__(physical_qubits, schedule, widths=widths, initial_state=initial_state,
                         meas_basis=meas_basis, time_unit=time_unit,
                         experiment_index=experiment_index, backend=backend)

        if control_state in [0, 1, 2]:
            self.set_experiment_options(control_state=control_state)
        else:
            raise ValueError(f'Invalid control state value {control_state}')

    def _pre_circuit(self) -> QuantumCircuit:
        circuit = super()._pre_circuit()

        if self.experiment_options.control_state >= 1:
            circuit.x(0)
        if self.experiment_options.control_state == 2:
            circuit.append(SetF12(), [0])
            circuit.append(X12Gate(), [0])

        circuit.metadata['control_state'] = self.experiment_options.control_state

        return circuit


def cr_rabi_init(control_state: int):
    return partial(CRRabi, control_state=control_state)
