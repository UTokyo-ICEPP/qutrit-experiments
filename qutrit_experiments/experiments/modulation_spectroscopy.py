from typing import List, Iterable, Optional, Sequence
from qiskit import QuantumCircuit, pulse
from qiskit.circuit import Gate, Parameter
from qiskit.providers import Backend
from qiskit.pulse import ScheduleBlock
from qiskit.qobj.utils import MeasLevel
from qiskit_experiments.curve_analysis import ResonanceAnalysis
from qiskit_experiments.framework import BackendData, BaseExperiment, Options

from ..calibrations.pulse_library import ModulatedGaussianSquare
from ..common.transpilation import replace_calibration_and_metadata


class ModulationSpectroscopy(BaseExperiment):
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.frequencies = None
        options.base_frequency = None
        options.amp = 0.02
        options.duration = 512
        options.sigma = 64
        options.width = 256
        return options

    @classmethod
    def _default_run_options(cls) -> Options:
        """Default options values for the experiment :meth:`run` method."""
        options = super()._default_run_options()

        options.meas_level = MeasLevel.KERNELED
        options.meas_return = "avg"

        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        frequencies: Iterable[float],
        base_frequency: Optional[float] = None,
        backend: Optional[Backend] = None,
    ):
        super().__init__(physical_qubits, analysis=ResonanceAnalysis(), backend=backend)

        self.set_experiment_options(frequencies=frequencies, base_frequency=base_frequency)

    def circuits(self) -> List[QuantumCircuit]:
        if self.backend is None:
            raise RuntimeError('ModulationSpectroscopy requires a backend to be set.')

        bdata = BackendData(self.backend)

        if self.experiment_options.base_frequency is None:
            base_frequency = bdata.drive_freqs[self.physical_qubits[0]]
        else:
            base_frequency = self.experiment_options.base_frequency

        frequency = Parameter('frequency')
        detuning = (frequency - base_frequency) * bdata.dt

        channel = pulse.DriveChannel(self.physical_qubits[0])

        with pulse.build(name='spectroscopy') as sched:
            pulse.set_frequency(base_frequency, channel)
            pulse.play(
                ModulatedGaussianSquare(
                    duration=self.experiment_options.duration,
                    amp=self.experiment_options.amp,
                    base_fraction=0.,
                    freq=detuning,
                    sigma=self.experiment_options.sigma,
                    width=self.experiment_options.width
                ),
                channel
            )

        gate = Gate('spectroscopy', 1, [frequency])

        template = QuantumCircuit(1)
        template.append(gate, [0])
        template.measure_active()

        template.add_calibration('spectroscopy', self.physical_qubits, sched, params=[frequency])

        template.metadata = {
            'qubits': self.physical_qubits
        }

        # Create the circuits to run
        circuits = []
        for freq in self.experiment_options.frequencies:
            circuit = template.assign_parameters({frequency: freq}, inplace=False)
            circuit.metadata['xval'] = freq
            circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        return replace_calibration_and_metadata(self.circuits(), self.physical_qubits,
                                                self.transpile_options.target)

    def _metadata(self):
        metadata = super()._metadata()
        # Store measurement level and meas return if they have been
        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)
        return metadata
