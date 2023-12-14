from typing import Sequence
import numpy as np
from uncertainties import UFloat, unumpy as unp
from qiskit.providers import Backend
from qiskit_experiments.framework import BackendData, ExperimentData


def compute_omega_0_angle(experiment_data: ExperimentData) -> UFloat:
    child_data = experiment_data.child_data(experiment_data.metadata["component_child_index"][0])
    # omega^0_g/2
    omega_0_components = child_data.analysis_results('hamiltonian_components').value
    return unp.arctan2(omega_0_components[1], omega_0_components[0]).item()


def find_target_angle(backend: Backend, qubits: Sequence[int]):
    # Perturbative approximation for qubit-qubit Hamiltonian components (Malekakhlagh et al. PRA 102):
    #   ωIx = -JΩ/(Δct+αc)
    #   ωzx = JΩ/(Δct+αc)-JΩ/Δct
    # ω0x = ωIx+ωzx = -JΩ/Δct is the control-0 Hamiltonian component also for a qutrit-qubit system
    drive_freqs = BackendData(backend).drive_freqs
    delta_ct = drive_freqs[qubits[0]] - drive_freqs[qubits[1]]

    return np.pi if np.sign(delta_ct) > 0. else 0.
