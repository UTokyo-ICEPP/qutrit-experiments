from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np
import lmfit
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.pulse import ScheduleBlock
from qiskit.providers import Backend
from qiskit_experiments.framework import (AnalysisResultData, BaseAnalysis, BaseExperiment,
                                          ExperimentData, Options)
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.data_processing import DataProcessor, Probability
from qiskit_experiments.visualization import CurvePlotter, MplDrawer

from ..common.data_processing import ToExpectation
from ..common.transpilation import map_to_physical_qubits
from .bloch import amp_func, phase_func, base_func


twopi = 2. * np.pi

class UnitaryTomography(BaseExperiment):
    r"""Determination of the gate unitary through repeated applications of the gate.

    Given a gate with unitary

    .. math::

        U = \exp\left(-\frac{i}{2} \theta [\sin\psi \cos\phi X + \sin\psi \sin\phi Y + \cos\psi Z]\right),

    the Bloch-sphere representation of :math:`U^n` is the rotation matrix

    .. math::

        R^n = \begin{pmatrix}
          (1 - \sin^2 \psi^k \cos^2 \phi^k) \cos(n \theta) + \sin^2 \psi^k \cos^2 \phi^k
        & C^k \cos(n \theta - \gamma^k) + \sin^2 \psi^k \sin \phi^k \cos \phi^k
        & A^k \cos(n \theta + \alpha^k) + \sin \psi^k \cos \psi^k \cos \phi^k \\
          C^k \cos(n \theta + \gamma^k) + \sin^2 \psi^k \sin \phi^k \cos \phi^k
        & (1 - \sin^2 \psi^k \sin^2 \phi^k) \cos(n \theta) + \sin^2 \psi^k \sin^2 \phi^k
        & B^k \cos(n \theta + \beta^k) + \sin \psi^k \cos \psi^k \sin \phi^k \\
          A^k \cos (n \theta - \alpha^k) + \sin \psi^k \cos \psi^k \cos \phi^k
        & B^k \cos(n \theta - \beta^k) + \sin \psi^k \cos \psi^k \sin \phi^k
        & \sin^2 \psi^k \cos(n \theta) + \cos^2 \psi^k
        \end{pmatrix},

    where

    .. math::

        A^k & = \sin \psi^k \sqrt{\cos^2 \psi^k \cos^2 \phi^k + \sin^2 \phi^k}, \\
        B^k & = \sin \psi^k \sqrt{\cos^2 \psi^k \sin^2 \phi^k + \cos^2 \phi^k}, \\
        C^k & = \sqrt{\sin^4 \psi^k \sin^2 \phi^k \cos^2 \phi^k + \cos^2 \psi^k}

    and

    .. math::

        \cos \alpha^k & = -\sin \psi^k \cos \psi^k \cos \phi^k / A^k, \\
        \sin \alpha^k & = -\sin \psi^k \sin \phi^k / A^k, \\
        \cos \beta^k & = -\sin \psi^k \cos \psi^k \sin \phi^k / B^k, \\
        \sin \beta^k & = \sin \psi^k \cos \phi^k / B^k, \\
        \cos \gamma^k & = -\sin^2 \psi^k \sin \phi^k \cos \phi^k / C^k, \\
        \sin \gamma^k & = -\cos \psi^k / C^k.
    """
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.schedule = None
        options.pre_circuit = None
        options.repetitions = np.arange(1, 11)
        options.measured_logical_qubit = 0
        options.invert_x_sign = False
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        schedule: ScheduleBlock,
        backend: Optional[Backend] = None,
        repetitions: Optional[Sequence[int]] = None,
        measured_logical_qubit: Optional[int] = None
    ):
        super().__init__(physical_qubits, backend=backend, analysis=UnitaryTomographyAnalysis())
        self.set_experiment_options(schedule=schedule)
        if repetitions is not None:
            self.set_experiment_options(repetitions=repetitions)
        if measured_logical_qubit is not None:
            self.set_experiment_options(measured_logical_qubit=measured_logical_qubit)

        self.extra_metadata = {}

    def circuits(self) -> List[QuantumCircuit]:
        options = self.experiment_options
        measured_qubit = options.measured_logical_qubit

        template = QuantumCircuit(len(self.physical_qubits), 1)
        if isinstance(options.pre_circuit, QuantumCircuit):
            template.compose(options.pre_circuit, inplace=True)
            template.barrier()
        template.add_calibration('unitary', self.physical_qubits[0:1], options.schedule)
        template.metadata = {
            'experiment_type': self._type,
            'qubits': self.physical_qubits,
            'readout_qubits': [self.physical_qubits[measured_qubit]]
        }

        circuits = []
        for meas_basis in ['x', 'y', 'z']:
            for nrep in options.repetitions:
                circuit = template.copy()

                for _ in range(nrep):
                    circuit.append(Gate('unitary', 1, []), [0])

                circuit.barrier()

                if meas_basis == 'x':
                    # Hadamard with global phase pi/4 sans last rz
                    angle = np.pi / 2.
                    if options.invert_x_sign:
                        angle *= -1.
                    circuit.rz(angle, measured_qubit)
                    circuit.sx(measured_qubit)
                elif meas_basis == 'y':
                    # Sdg + Hadamard sans last rz
                    circuit.sx(measured_qubit)

                circuit.measure(measured_qubit, 0)

                circuit.metadata.update({
                    'meas_basis': meas_basis,
                    'xval': int(nrep)
                })

                circuits.append(circuit)

        return circuits

    def _transpiled_circuits(self) -> List[QuantumCircuit]:
        return [map_to_physical_qubits(circuit, self.physical_qubits,
                                       self.transpile_options.target)
                for circuit in self.circuits()]

    def _metadata(self) -> Dict[str, Any]:
        metadata = super()._metadata()

        # set for the experiment
        for run_opt in ["meas_level", "meas_return"]:
            if hasattr(self.run_options, run_opt):
                metadata[run_opt] = getattr(self.run_options, run_opt)

        metadata.update(self.extra_metadata)

        return metadata


class UnitaryTomographyAnalysis(curve.CurveAnalysis):
    @classmethod
    def _default_options(cls):
        options = super()._default_options()
        options.outcome = '0'
        options.data_processor = DataProcessor('counts', [Probability('0'), ToExpectation()])
        return options

    @staticmethod
    def func_meas_x(x, theta, psi, phi):
        amp = amp_func(6, psi, phi)
        phase = phase_func(6, psi, phi, 0.)
        base = base_func(6, psi, phi)
        return amp * np.cos(x * theta + phase) + base

    @staticmethod
    def func_meas_y(x, theta, psi, phi):
        amp = amp_func(7, psi, phi)
        phase = phase_func(7, psi, phi, 0.)
        base = base_func(7, psi, phi)
        return amp * np.cos(x * theta + phase) + base

    @staticmethod
    def func_meas_z(x, theta, psi, phi):
        amp = amp_func(8, psi, phi)
        phase = phase_func(8, psi, phi, 0.)
        base = base_func(8, psi, phi)
        return amp * np.cos(x * theta + phase) + base

    def __init__(self, name: Optional[str] = None):
        super().__init__(
            models=[
                lmfit.Model(self.func_meas_x, name='x'),
                lmfit.Model(self.func_meas_y, name='y'),
                lmfit.Model(self.func_meas_z, name='z'),
            ],
            name=name
        )
        self.set_options(
            data_subfit_map={basis: {'meas_basis': basis} for basis in ['x', 'y', 'z']},
            bounds={
                'psi': (-1.e-3, np.pi + 1.e-3),
                'phi': (-twopi, twopi),
                'theta': (-1.e-3, twopi + 1.e-3)
            }
        )

    def _generate_fit_guesses(
        self,
        user_opt: curve.FitOptions,
        curve_data: curve.CurveData,
    ) -> Union[curve.FitOptions, List[curve.FitOptions]]:
        labels = ['x', 'y', 'z']

        subdata = {label: curve_data.get_subset_of(label) for label in labels}
        bases = {label: curve.guess.constant_sinusoidal_offset(subdata[label].y) for label in labels}

        phi = np.arctan2(bases['y'], bases['x'])
        sphi = np.sin(phi)
        if not np.isclose(sphi, 0.):
            spsicpsi = bases['y'] / sphi
        else:
            spsicpsi = bases['x'] / np.cos(phi)
        psi = np.arctan2(spsicpsi, bases['z'])

        user_opt.p0.set_if_empty(psi=psi, phi=phi)

        options = []
        for label in ['x', 'y', 'z']:
            new_opt = user_opt.copy()
            new_opt.p0.set_if_empty(
                theta=curve.guess.frequency(subdata[label].x, subdata[label].y) * twopi
            )
            options.append(new_opt)

        return options

    def _run_curve_fit(
        self,
        curve_data: curve.CurveData,
        models: List[lmfit.Model],
    ) -> curve.CurveFitResult:
        result = super()._run_curve_fit(curve_data, models)

        if result.success:
            while result.params['phi'] < -np.pi:
                result.params['phi'] += twopi
            while result.params['phi'] > np.pi:
                result.params['phi'] -= twopi

        return result
