# amg_pcg.py — minimal subset required by magnetostatics.py

import numpy as np
import scipy.sparse as sp


__all__ = ["_assemble_scalar_laplacian"]


def _assemble_scalar_laplacian(
    N: int,
    conn: np.ndarray,       # (E, 4) int
    grad_phi: np.ndarray,   # (E, 4, 3) float64
    volume: np.ndarray,     # (E,) float64
) -> sp.csr_matrix:
    """
    Assemble the standard linear-tetrahedron scalar Laplacian stiffness matrix:

        L_ij = sum_e ∫_{Ω_e} (∇φ_i · ∇φ_j) dV
             ≈ sum_e V_e * (∇φ_i · ∇φ_j)

    Parameters
    ----------
    N : int
        Number of nodes (global dofs).
    conn : (E, 4) ndarray of int
        Element-to-node connectivity (tetrahedra with 4 local nodes).
    grad_phi : (E, 4, 3) ndarray of float64
        Gradients of the four linear basis functions in each element.
    volume : (E,) ndarray of float64
        Element volumes.

    Returns
    -------
    L : (N, N) scipy.sparse.csr_matrix
        Global scalar Laplacian matrix in CSR format.
    """
    # Element stiffness Ke[e,i,j] = V[e] * sum_k grad_phi[e,i,k] * grad_phi[e,j,k]
    Ke = (volume[:, None, None]) * np.einsum("eik,ejk->eij", grad_phi, grad_phi)  # (E,4,4)

    # Expand connectivity into all (i,j) pairs per element
    rows = np.repeat(conn, 4, axis=1).reshape(-1)   # (E*16,)
    cols = np.tile(conn, (1, 4)).reshape(-1)        # (E*16,)
    data = Ke.reshape(-1)                            # (E*16,)

    L = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()
    return L
