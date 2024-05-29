"""Fine amplitude calibration for X12 and SX12 gates."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.providers import Backend
from qiskit_experiments.framework import Options
from qiskit_experiments.calibration_management import Calibrations
from qiskit_experiments.library import FineAmplitude, FineAmplitudeCal

from ...calibrations import get_qutrit_pulse_gate
from ...experiment_mixins.ef_space import EFSpaceExperiment
from ...gates import X12Gate, SX12Gate
from ...transpilation import map_to_physical_qubits


class CustomTranspiledFineAmplitude(FineAmplitude):
    """FineAmplitude with optimized transpiler sequence."""
    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        repetitions = self.experiment_options.repetitions
        gate = self.experiment_options.gate
        circuits = self.circuits()
        icirc = 0

        transpiled_circuits = []

        if self.experiment_options.add_cal_circuits:
            transpiled_circuits.extend(
                map_to_physical_qubits(circuit, self.physical_qubits,
                                       self._backend_data.coupling_map)
                for circuit in circuits[:2]
            )
            icirc = 2

        first_circuit = map_to_physical_qubits(circuits[icirc], self.physical_qubits,
                                               self._backend_data.coupling_map)
        transpiled_circuits.append(first_circuit)

        gate_position = next(pos for pos, inst in enumerate(first_circuit.data)
                             if inst.operation.name == gate.name)

        for circuit, repetition in enumerate(circuits[icirc + 1:], repetitions[1:]):
            tcirc = first_circuit.copy()
            for _ in range(repetition - repetitions[0]):
                tcirc.data.insert(gate_position,
                                  CircuitInstruction(gate, [self.physical_qubits[0]]))
            tcirc.metadata = circuit.metadata
            transpiled_circuits.append(tcirc)

        return transpiled_circuits


class EFFineAmplitude(EFSpaceExperiment, CustomTranspiledFineAmplitude):
    """EF-space FineAmplitude."""


class EFFineAmplitudeCal(FineAmplitudeCal, EFFineAmplitude):
    """Calibration experiment for EFFineAmplitude."""
    def _attach_calibrations(self, circuit: QuantumCircuit):
        for gate in ['x12', 'sx12']:
            sched = get_qutrit_pulse_gate(gate, self.physical_qubits[0], self._cals,
                                          target=self._backend.target)
            circuit.add_calibration(gate, self.physical_qubits, sched)


class EFFineXAmplitudeCal(EFFineAmplitudeCal):
    """Specialization of EFFineAmplitude for X12."""
    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            'x12',
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            gate=X12Gate()
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi,
                "phase_offset": np.pi / 2,
            }
        )

    def _pre_circuit(self, num_clbits: int) -> QuantumCircuit:
        """The preparation circuit is an sx gate to move to the equator of the Bloch sphere."""
        circuit = super()._pre_circuit(num_clbits)
        circuit.append(SX12Gate(), qargs=[0])
        return circuit


class EFFineSXAmplitudeCal(EFFineAmplitudeCal):
    """Specialization of EFFineAmplitude for SX12."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        r"""Default values for the fine amplitude experiment.
        Experiment Options:
            add_sx (bool): This option is False by default when calibrating gates with a target
                angle per gate of :math:`\pi/2` as this increases the sensitivity of the
                experiment.
            add_xp_circuit (bool): This option is False by default when calibrating gates with
                a target angle per gate of :math:`\pi/2`.
            repetitions (list[int]): By default the repetitions take on odd numbers for
                :math:`\pi/2` target angles as this ideally prepares states on the equator of
                the Bloch sphere. Note that the repetitions include two repetitions which
                plays the same role as including a circuit with an X gate.
            target_angle (float): The target angle per gate.
        """
        options = super()._default_experiment_options()
        options.add_cal_circuits = False
        options.repetitions = [0, 1, 2, 3, 5, 7, 9, 11, 13, 15, 17, 21, 23, 25]
        options.target_angle = np.pi / 2
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        calibrations: Calibrations,
        backend: Optional[Backend] = None,
        cal_parameter_name: Optional[str] = "amp",
        auto_update: bool = True,
    ):
        super().__init__(
            physical_qubits,
            calibrations,
            'sx12',
            backend=backend,
            cal_parameter_name=cal_parameter_name,
            auto_update=auto_update,
            gate=SX12Gate()
        )
        self.analysis.set_options(
            fixed_parameters={
                "angle_per_gate": np.pi / 2,
                "phase_offset": np.pi,
            }
        )
