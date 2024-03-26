"""Theoretical predictions for siZZle."""
from collections.abc import Sequence
from typing import Optional
import numpy as np

twopi = 2. * np.pi


def perturbed_eigvals(hamiltonian: np.ndarray) -> np.ndarray:
    """Return the energy eigenvalues of a near-diagonal hermitian matrix.

    Args:
        hamiltonian: An array representing (a) hermitian matrix(ces) whose last two dimensions
            have shape (n, n).

    Returns:
        An array representing the energy eigenvalues. For ``hamiltonian`` of shape (..., n, n), the
        shape of the returned array is (..., n).
    """
    energies, unitary = np.linalg.eigh(hamiltonian)
    k = np.argmax(np.abs(unitary), axis=-1)
    # Reordered eigenvalues (makes the corresponding change-of-basis unitary closest to identity)
    if len(k.shape) > 1:
        # Indexing trick: Index into a flattened energies array to sort just the last dim
        # I think there is a better way to do the same thing but can't think of one right now
        k += np.arange(np.prod(k.shape[:-1]))[:, None] * k.shape[-1]
    return energies.reshape(-1)[k.reshape(-1)].reshape(energies.shape)


def diag_paulis(nlevels: int):
    paulis = np.zeros((nlevels, nlevels), dtype=float)
    paulis[0] = np.sqrt(2. / nlevels)
    for ilevel in range(1, nlevels):
        paulis[ilevel, :ilevel] = 1.
        paulis[ilevel, ilevel] = -ilevel
        paulis[ilevel] *= np.sqrt(2. / (ilevel + ilevel**2))

    return paulis


def diag_quditops(nlevels: int):
    ops = np.zeros((nlevels, nlevels), dtype=float)
    ops[0] = 1.
    for ilevel in range(1, nlevels):
        ops[ilevel, ilevel - 1] = 1.
        ops[ilevel, ilevel] = -1.

    return ops


def get_qudit_components(diagonals: np.ndarray, truncate_to: Optional[tuple[int, int]] = None):
    """Extract the Qudit (Izζ)-basis components of the diaonal Hamiltonian.

    The function first extracts the (generalized) Pauli decomposition of the diagonals array by
    taking the inner products with Pauli products. The extracted Pauli components are then
    transformed to qudit (non-orthogonal) basis.

    Args:
        diagonals: An array representing e.g. energy eigenvalues. If a multi-dimensional array is
            given, the last dimension is considered to represent the diagonals, and the remaining
            dimensions are kept in the returned array.
        truncate_to: Truncate the qudit-basis components to (control_levels, target_levels).

    Returns:
        For an input array of shape (..., n**2), an array of shape (..., n, n). If ``truncate_to``
        is not None, the shape is ``(...,) + truncate_to``.
    """
    nlevels = int(np.round(np.sqrt(diagonals.shape[-1])))
    if truncate_to is None:
        paulis = (diag_paulis(nlevels),) * 2
    else:
        if truncate_to[0] > nlevels or truncate_to[1] > nlevels:
            raise ValueError(f'Cannot truncate matrix of shape {diagonals.shape} to {truncate_to}')

        paulis = tuple(diag_paulis(dim) for dim in truncate_to)
        input_trunc = diagonals.reshape(diagonals.shape[:-1] + (nlevels, nlevels))
        input_trunc = input_trunc[..., :truncate_to[0], :truncate_to[1]]
        diagonals = input_trunc.reshape(diagonals.shape[:-1] + (np.prod(truncate_to),))

    pauli_prods = np.kron(*paulis).reshape(paulis[0].shape[0], paulis[1].shape[0], -1)
    if len(diagonals.shape) > 1:
        pauli_prods = np.tile(pauli_prods, diagonals.shape[:-1] + (1, 1, 1))

    # Shape (..., nlevels, nlevels)
    pauli_components = np.sum(pauli_prods * diagonals[..., None, None, :], axis=-1) / 4.

    # Transform the control components to qudit basis
    qudit_basis = tuple(diag_quditops(p.shape[0]) for p in paulis)
    # Qudit basis
    #  T = [I, z, ζ, ...]
    # Pauli basis
    #  λ = [λ0, λ3, λ8, ...]
    # Transformation matrix R is
    #  T = Rλ
    tr_mats = tuple(np.sum(q[:, None, :] * p, axis=-1) / 2. for q, p in zip(qudit_basis, paulis))
    # Diagonals = [(pauli_components) λ] x [(pauli_components) λ]
    #           = [(qudit_components) T] x [(qudit_components) T]
    # -> (qudit_components) = (pauli_components) R^(-1)
    qudit_components = np.tensordot(pauli_components, np.linalg.inv(tr_mats[0]), (-2, 0))
    # Original dimension -1 is now moved to -2
    qudit_components = np.tensordot(qudit_components, np.linalg.inv(tr_mats[1]), (-2, 0))

    return qudit_components


def free_hamiltonian(
    hvars: dict[str, float],
    qubits: tuple[int, int],
    nlevels: int = 3
) -> np.ndarray:
    """Construct the free Hamiltonian of a given qubit pair."""
    qudit_energies = []
    for qubit in qubits:
        energies = np.arange(nlevels) * hvars[f'wq{qubit}']
        energies += np.concatenate([[0.], np.cumsum(np.arange(nlevels - 1))]) * hvars[f'delta{qubit}']
        qudit_energies.append(energies)

    # Free Hamiltonian eigenvalues
    efree = np.add.outer(qudit_energies[0], qudit_energies[1]).flatten()

    return np.diagflat(efree)


def static_hamiltonian(
    hvars: dict[str, float],
    qubits: tuple[int, int],
    nlevels: int = 3
) -> np.ndarray:
    """Construct the static Hamiltonian of a given qubit pair.

    Args:
        hvars: ``backend.configuration().hamiltonian['vars']``
        qubits: (control_qubit, target_qubit)
        nlevels: Number of levels to consider in calculation.

    Returns:
        A square array with shape ``(nlevels**2, nlevels**2)`` that represents the lab-frame static
        Hamiltonian in the free Hamiltonian energy eigenbasis.
    """
    if (coupling := hvars.get(f'jq{min(qubits)}q{max(qubits)}', 0.)) == 0.:
        # Sometimes (e.g. kawasaki) the couplings have not been measured
        coupling = 11.e+6

    annihilation = np.diagflat(np.sqrt(np.arange(1, nlevels)), 1)
    creation = np.diagflat(np.sqrt(np.arange(1, nlevels)), -1)

    hstat = free_hamiltonian(hvars, qubits, nlevels=nlevels)
    hstat += coupling * np.kron(creation, annihilation)
    hstat += coupling * np.kron(annihilation, creation)

    return hstat


def sizzle_shifted_energies(
    hvars: dict[str, float],
    qubits: tuple[int, int],
    amps: tuple[float, float],
    frequency: 'array_like',
    phase: float = 0.,
    nlevels: int = 3
) -> np.ndarray:
    """Calculate the AC Stark shifted Hamiltonian eigenvalues.

    When two coupled qubits are driven at a common frequency, the static Hamiltonian in the drive
    frame is still static because of the frequency cancellation between the qubits.

    Args:
        hvars: ``backend.configuration().hamiltonian['vars']``
        qubits: (control_qubit, target_qubit)
        amps: (control_amp, target_amp) relative to ``hvars['omegadX']``
        frequency: siZZle drive frequency in Hz. Can have additional dimensions in front.
        nlevels: Number of levels to consider.

    Returns:
        An array of shape (..., nlevels) where ... corresponds to the additional dimensions of
        ``frequency``.
    """
    physical_amps = tuple(a * hvars[f'omegad{q}'] for a, q in zip(amps, qubits))
    omega_drive = np.asarray(frequency) * twopi

    hstat = static_hamiltonian(hvars, qubits, nlevels=nlevels).astype(complex)

    stair = np.add.outer(np.arange(nlevels), np.arange(nlevels)).flatten().astype(float)
    drive_diag = np.tile(stair, omega_drive.shape + (1,))
    drive_diag *= omega_drive[..., None]

    identity = np.diagflat(np.ones(nlevels, dtype=complex))
    annihilation = np.diagflat(np.sqrt(np.arange(1, nlevels, dtype=complex)), 1)
    creation = np.diagflat(np.sqrt(np.arange(1, nlevels, dtype=complex)), -1)

    # Drive-frame Hamiltonian: Coupling terms are identical to lab frame
    hfull_driveframe = np.tile(hstat, omega_drive.shape + (1, 1))
    # Subtract the drive term from diagonal
    hfull_driveframe[..., np.arange(nlevels ** 2), np.arange(nlevels ** 2)] -= drive_diag
    # Add the drive terms
    creation_ph = np.exp(-1.j * phase) * creation
    annihilation_ph = np.exp(1.j * phase) * annihilation
    hfull_driveframe += 0.5 * physical_amps[0] * np.kron(creation_ph + annihilation_ph, identity)
    hfull_driveframe += 0.5 * physical_amps[1] * np.kron(identity, creation + annihilation)

    # Drive-frame eigenvalues + subtracted diagonals
    return perturbed_eigvals(hfull_driveframe).real + drive_diag


def sizzle_hamiltonian_shifts(
    hvars: dict[str, float],
    qubits: tuple[int, int],
    amps: tuple[float, float],
    frequency: 'array_like',
    phase: float = 0.,
    nlevels: int = 3,
    truncate_to: tuple[int, int] = (3, 2)
):
    hstat = static_hamiltonian(hvars, qubits, nlevels=nlevels)
    efull = sizzle_shifted_energies(hvars, qubits, amps, frequency, phase=phase, nlevels=nlevels)
    return get_qudit_components(efull - perturbed_eigvals(hstat), truncate_to=truncate_to)
