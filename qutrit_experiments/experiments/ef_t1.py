"""A T1 experiment for qutrits."""
from collections.abc import Sequence
from typing import Optional, Union
import lmfit
import numpy as np
import scipy.linalg as scilin
from uncertainties import unumpy as unp
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, CircuitInstruction
from qiskit.circuit.library import XGate
from qiskit.providers import Backend
from qiskit_experiments.data_processing import DataProcessor
from qiskit_experiments.data_processing.exceptions import DataProcessorError
import qiskit_experiments.curve_analysis as curve
from qiskit_experiments.framework import ExperimentData, Options
from qiskit_experiments.library import T1

from ..data_processing import MultiProbability, ReadoutMitigation, SerializeMultiProbability
from ..gates import X12Gate


class EFT1(T1):
    """A T1 experiment for qutrits."""
    @classmethod
    def _default_experiment_options(cls) -> Options:
        options = super()._default_experiment_options()
        options.delays = np.linspace(0., 5.e-4, 30)
        # Number of empty circuits to insert to let the |2> state relax
        options.insert_empty_circuits = 0
        return options

    def __init__(
        self,
        physical_qubits: Sequence[int],
        delays: Optional[Union[list[float], np.array]] = None,
        backend: Optional[Backend] = None
    ):
        if (delays_exp := delays) is None:
            delays_exp = self._default_experiment_options().delays

        super().__init__(physical_qubits, delays_exp, backend=backend)
        self.analysis = EFT1Analysis()

        if delays is not None:
            self.set_experiment_options(delays=delays)

    def circuits(self) -> list[QuantumCircuit]:
        sup_circuits = super().circuits()
        measure_0 = QuantumCircuit(1)
        measure_0.measure_all()
        circuits = []
        for circuit in sup_circuits:
            for idx, inst in enumerate(circuit.data):
                if isinstance(inst.operation, XGate):
                    circuit.data.insert(idx + 1,
                                        CircuitInstruction(X12Gate(), [self.physical_qubits[0]]))
            circuits.append(circuit)
            for _ in range(self.experiment_options.insert_empty_circuits):
                circuits.append(measure_0)

        return circuits

    def _transpiled_circuits(self) -> list[QuantumCircuit]:
        circuits = super()._transpiled_circuits()
        for circuit in circuits:
            creg = ClassicalRegister(size=2)
            circuit.remove_final_measurements(inplace=True)
            circuit.add_register(creg)
            # Post-transpilation - logical qubit = physical qubit
            circuit.measure(self.physical_qubits[0], creg[0])
            circuit.x(self.physical_qubits[0])
            circuit.measure(self.physical_qubits[0], creg[1])

        return circuits


def _three_component_relaxation(time, p0, p1, p2, g10, g20, g21):
    transition_matrix = np.array(
        [[0., g10, g20],
        [0., -g10, g21],
        [0., 0., -(g20 + g21)]]
    )
    return scilin.expm(time * transition_matrix) @ np.array([p0, p1, p2])

def _prob_0(time, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(time, p0, p1, p2, g10, g20, g21)[0]

def _prob_1(time, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(time, p0, p1, p2, g10, g20, g21)[1]

def _prob_2(time, p0, p1, p2, g10, g20, g21):
    return _three_component_relaxation(time, p0, p1, p2, g10, g20, g21)[2]


class EFT1Analysis(curve.CurveAnalysis):
    """Run fit with rate equations."""

    @classmethod
    def _default_options(cls) -> Options:
        options = super()._default_options()
        options.assignment_matrix = None
        return options

    def __init__(self, name: Optional[str] = None):
        super().__init__(models=[
            lmfit.Model(_prob_0, name='0'),
            lmfit.Model(_prob_1, name='1'),
            lmfit.Model(_prob_2, name='2')
        ], name=name)

        self.options.result_parameters = [
            curve.ParameterRepr('g10', 'Γ10'),
            curve.ParameterRepr('g20', 'Γ20'),
            curve.ParameterRepr('g21', 'Γ21')
        ]

        self.options.plotter.set_figure_options(
            xlabel="Delay",
            ylabel="Probability",
            xval_unit="s",
        )

    def _initialize(self, experiment_data: ExperimentData):
        if self.options.data_processor is None:
            nodes = []
            if (cal_matrix := self.options.assignment_matrix) is not None:
                nodes.append(ReadoutMitigation(cal_matrix))
            nodes += [
                MultiProbability(),
                SerializeMultiProbability(['10', '01', '11'])
            ]
            self.options.data_processor = DataProcessor('counts', nodes)

        super()._initialize(experiment_data)

    def _run_data_processing(
        self,
        raw_data: list[dict],
        models: list[lmfit.Model]
    ) -> curve.CurveData:
        x_key = self.options.x_key
        try:
            xdata = np.asarray([datum["metadata"][x_key] for datum in raw_data], dtype=float)
        except KeyError as ex:
            raise DataProcessorError(
                f"X value key {x_key} is not defined in circuit metadata."
            ) from ex

        # ydata is serialized multiprobability
        ydata = self.options.data_processor(raw_data)
        data_allocation = np.tile(['0', '1', '2'], xdata.shape[0])
        shots = np.repeat([datum.get("shots", np.nan) for datum in raw_data], 3)
        xdata = np.repeat(xdata, 3)

        return curve.CurveData(
            x=xdata,
            y=unp.nominal_values(ydata),
            y_err=unp.std_devs(ydata),
            shots=shots,
            data_allocation=data_allocation,
            labels=['0', '1', '2']
        )
