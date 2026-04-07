import jax
import jax.numpy as jnp
from typing import Union
from geom import TetGeom

# Vacuum permeability (H/m)
MU0 = 4e-7 * jnp.pi

def magnetic_volume(
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
    Ms_tol: float = 0.0,
) -> jnp.ndarray:
    """
    Total magnetic volume: sum of element volumes where Ms > Ms_tol.
    """
    g_ids = geom.mat_id - 1                  # (E,)
    Ms_e = jnp.asarray(Ms_lookup)[g_ids]     # (E,)
    mask = Ms_e > Ms_tol
    return jnp.sum(geom.volume * mask)

# ----------------------------
# Exchange (value + gradient)
# ----------------------------
def exchange_energy_and_grad(
    m_nodes: jnp.ndarray,        # (N, 3)
    geom: TetGeom,               # conn:(E,4); grad_phi:(E,4,3); volume:(E,); mat_id:(E,)
    A_lookup: jnp.ndarray,       # (G,) exchange stiffness [J/m]
):
    conn, grad_phi, Ve, mat_id = geom.conn, geom.grad_phi, geom.volume, geom.mat_id
    g_ids = mat_id - 1
    A_e = A_lookup[g_ids]        # (E,)
    m_e = m_nodes[conn]          # (E, 4, 3)
    # G_e[l,k] = sum_alpha m_e[alpha,l] * grad_phi[alpha,k]  -> (E,3,3)
    G_e = jnp.einsum('eal,eak->elk', m_e, grad_phi)
    # Energy
    E_e = A_e * Ve * jnp.sum(G_e**2, axis=(1, 2))
    E = jnp.sum(E_e)
    # Gradient assembly (scatter-add back to nodes)
    # dot_term[e, gamma, l] = sum_k G_e[e, l, k] * grad_phi[e, gamma, k]
    dot_term = jnp.einsum('elk,egk->egl', G_e, grad_phi)    # (E,4,3)
    contrib = (2.0 * A_e * Ve)[:, None, None] * dot_term    # (E,4,3)
    grad_m = jnp.zeros_like(m_nodes).at[conn].add(contrib)  # (N,3)
    return E, grad_m

# ---------------------------------
# Uniaxial anisotropy (value + grad)
# ---------------------------------
def uniaxial_anisotropy_energy_and_grad(
    m_nodes: jnp.ndarray,    # (N, 3)
    geom: TetGeom,           # conn:(E,4), volume:(E,), mat_id:(E,)
    K1_lookup: jnp.ndarray,  # (G,)
    k_easy_e: jnp.ndarray,   # (E, 3) unit easy axis per element
):
    conn, Ve, mat_id = geom.conn, geom.volume, geom.mat_id
    g_ids = mat_id - 1
    K1_e = K1_lookup[g_ids]                   # (E,)
    m_e = m_nodes[conn]                       # (E, 4, 3)
    v_e = jnp.einsum('eac,ec->ea', m_e, k_easy_e)  # (E, 4)
    # Energy
    sum_v = jnp.sum(v_e, axis=1)             # (E,)
    sum_v2 = jnp.sum(v_e**2, axis=1)         # (E,)
    quad = (sum_v**2 + sum_v2) * (Ve / 20.0) # (E,)
    E_e = -K1_e * quad
    E = jnp.sum(E_e)
    # Gradient
    Mv = (Ve / 20.0)[:, None] * (sum_v[:, None] + v_e)   # (E,4)
    factor = (-2.0 * K1_e)[:, None] * Mv                 # (E,4)
    contrib = factor[..., None] * k_easy_e[:, None, :]   # (E,4,3)
    grad_m = jnp.zeros_like(m_nodes).at[conn].add(contrib)
    return E, grad_m

# ---------------------------------
# Zeeman (uniform H) (value + grad)
# ---------------------------------
def zeeman_energy_uniform_field_and_grad(
    m_nodes: jnp.ndarray,
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
    H: jnp.ndarray,             # (3,)
):
    """
    Zeeman energy with spatially uniform external field H (shape (3,)):
      E_Z = - μ0 ∑_e Ms_e * V_e * (1/4) ∑_α m_e[α] · H
    Returns (E_Z, grad_m).
    """
    conn, Ve, mat_id = geom.conn, geom.volume, geom.mat_id
    H = jnp.asarray(H, dtype=m_nodes.dtype)             # (3,)
    g_ids = mat_id - 1
    Ms_e = Ms_lookup[g_ids]                             # (E,)
    m_e = m_nodes[conn]                                 # (E,4,3)
    sum_m = jnp.sum(m_e, axis=1)                        # (E,3)
    sum_m_dot_H = jnp.einsum('ec,c->e', sum_m, H)       # (E,)
    E_e = -MU0 * Ms_e * Ve * 0.25 * sum_m_dot_H
    E_Z = jnp.sum(E_e)
    # Gradient: same constant per local node inside each element
    scale = (-MU0 * Ms_e * Ve / 4.0)[:, None, None]     # (E,1,1)
    one_loc = scale * H[None, None, :]                  # (E,1,3)
    contrib = jnp.broadcast_to(one_loc, (conn.shape[0], 4, 3))
    grad_m = jnp.zeros_like(m_nodes).at[conn].add(contrib)
    return E_Z, grad_m

# ----------------------------------------------------------
# Brown functional S[m, A] (value + grads wrt m and A) (Eq. 80)
# ----------------------------------------------------------
def _element_curl_A(A_nodes: jnp.ndarray, conn: jnp.ndarray, grad_phi: jnp.ndarray) -> jnp.ndarray:
    """
    curl(A) is constant on each linear tet element.
    Returns curl_e with shape (E, 3).
    """
    A_e = A_nodes[conn]                                    # (E,4,3)
    # gradA[e, comp, deriv] = sum_alpha A_e[e,alpha,comp] * grad_phi[e,alpha,deriv]
    gradA = jnp.einsum('eac,eak->eck', A_e, grad_phi)      # (E,3,3)
    cx = gradA[:, 2, 1] - gradA[:, 1, 2]
    cy = gradA[:, 0, 2] - gradA[:, 2, 0]
    cz = gradA[:, 1, 0] - gradA[:, 0, 1]
    return jnp.stack([cx, cy, cz], axis=-1)                # (E,3)

def brown_energy_and_grad_from_m(
    m_nodes: jnp.ndarray,   # (N,3)
    A_nodes: jnp.ndarray,   # (N,3)
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
):
    """
    Single-pass evaluation of Brown's functional S[m, A] and its gradients.
    Returns: S (scalar), grad_m (N,3), grad_A (N,3)
    """
    conn, grad_phi, Ve, mat_id = geom.conn, geom.grad_phi, geom.volume, geom.mat_id

    # Material Ms per element
    g_ids = mat_id - 1
    Ms_e = Ms_lookup[g_ids]                                 # (E,)

    # Element data
    curl_e = _element_curl_A(A_nodes, conn, grad_phi)       # (E,3)
    m_e = m_nodes[conn]                                     # (E,4,3)
    M_e = Ms_e[:, None, None] * m_e                         # (E,4,3)
    sum_M = jnp.sum(M_e, axis=1)                            # (E,3)
    sum_M2 = jnp.sum(jnp.sum(M_e**2, axis=2), axis=1)       # (E,)

    # Energy terms (Eq. 80)
    E1 = jnp.sum((Ve / (2.0 * MU0)) * jnp.sum(curl_e**2, axis=1))
    E2 = jnp.sum(-(Ve / 4.0) * jnp.sum(sum_M * curl_e, axis=1))
    E3 = jnp.sum((MU0 / 2.0) * (Ve / 20.0) * (jnp.sum(sum_M**2, axis=1) + sum_M2))
    S = E1 + E2 + E3

    # Gradients w.r.t. m (via M = Ms * m)
    gM_term2 = -(Ve[:, None, None] / 4.0) * curl_e[:, None, :]                # (E,4,3)
    gM_term3 = MU0 * (Ve[:, None, None] / 20.0) * (sum_M[:, None, :] + M_e)   # (E,4,3)
    gM_e = gM_term2 + gM_term3                                                # (E,4,3)
    gm_e = Ms_e[:, None, None] * gM_e
    grad_m = jnp.zeros_like(m_nodes).at[conn].add(gm_e)

    # Gradients w.r.t. A
    gx, gy, gz = grad_phi[..., 0], grad_phi[..., 1], grad_phi[..., 2]         # (E,4)
    Jx = jnp.stack([jnp.zeros_like(gx), gz, -gy], axis=-1)                    # d curl / d A_x
    Jy = jnp.stack([-gz, jnp.zeros_like(gx), gx], axis=-1)                    # d curl / d A_y
    Jz = jnp.stack([gy, -gx, jnp.zeros_like(gx)], axis=-1)                    # d curl / d A_z

    dot_curl_Jx = jnp.einsum('ec,eac->ea', curl_e, Jx)
    dot_curl_Jy = jnp.einsum('ec,eac->ea', curl_e, Jy)
    dot_curl_Jz = jnp.einsum('ec,eac->ea', curl_e, Jz)

    dot_sumM_Jx = jnp.einsum('ec,eac->ea', sum_M, Jx)
    dot_sumM_Jy = jnp.einsum('ec,eac->ea', sum_M, Jy)
    dot_sumM_Jz = jnp.einsum('ec,eac->ea', sum_M, Jz)

    gA1_x = (Ve[:, None] / MU0) * dot_curl_Jx
    gA1_y = (Ve[:, None] / MU0) * dot_curl_Jy
    gA1_z = (Ve[:, None] / MU0) * dot_curl_Jz

    gA2_x = -(Ve[:, None] / 4.0) * dot_sumM_Jx
    gA2_y = -(Ve[:, None] / 4.0) * dot_sumM_Jy
    gA2_z = -(Ve[:, None] / 4.0) * dot_sumM_Jz

    gAx_e = gA1_x + gA2_x
    gAy_e = gA1_y + gA2_y
    gAz_e = gA1_z + gA2_z
    gA_e = jnp.stack([gAx_e, gAy_e, gAz_e], axis=-1)                          # (E,4,3)

    grad_A = jnp.zeros_like(A_nodes).at[conn].add(gA_e)
    return S, grad_m, grad_A

def brown_energy_and_grad_from_scalar_potential(
    m_nodes: jnp.ndarray,  # (N, 3) unit magnetization at nodes
    u_nodes: jnp.ndarray,  # (N,) scalar magnetic potential at nodes
    geom: TetGeom,         # conn:(E,4); grad_phi:(E,4,3); volume:(E,); mat_id:(E,)
    Ms_lookup: jnp.ndarray # (G,) spontaneous magnetization per material id
):
    """
    Brown's magnetostatic energy functional using the scalar potential u (Eq. 38):
        s[u; m] = - (mu0/2) ∫ |∇u|^2 dV + mu0 ∫_V M · ∇u dV,
    with h = -∇u and M = Ms * m.

    Discretization details (linear tetrahedra):
      - u(r) = Σ_α u_α φ_α,  ∇u is constant on each element: ∇u_e = Σ_α u_e[α] ∇φ_α
      - Element energy:
          s_e = -(mu0/2) * V_e * |∇u_e|^2
                + mu0 * Ms_e * V_e/4 * Σ_β u_e[β] * (Σ_α m_e[α] · ∇φ_β)
      - Element gradient wrt u (assemble to global):
          ∂s/∂u_e[β] = -mu0 * V_e * Σ_α (∇φ_α · ∇φ_β) u_e[α]
                        + mu0 * Ms_e * V_e/4 * Σ_α m_e[α] · ∇φ_β
      - Element gradient wrt m (each local node gets the same vector):
          ∂s/∂m_e[γ] = mu0 * Ms_e * V_e/4 * ∇u_e

    Returns:
      S        : scalar energy (same units as your other energy terms)
      grad_m   : (N, 3) gradient wrt nodal magnetization m
      grad_u   : (N,)   gradient wrt nodal scalar potential u

    Notes:
      - Matches the FEM expressions in (38), (43)–(47) for linear tets and midpoint rule.
      - Structure and assembly mirror `brown_energy_and_grad_from_m` for consistency.
    """
    conn, grad_phi, Ve, mat_id = geom.conn, geom.grad_phi, geom.volume, geom.mat_id
    # Element-wise material parameters
    g_ids = mat_id - 1
    Ms_e = Ms_lookup[g_ids]                     # (E,)

    # Gather per-element nodal values
    m_e = m_nodes[conn]                         # (E, 4, 3)
    u_e = u_nodes[conn]                         # (E, 4)

    # ∇u is constant over a linear tet:
    # gradU_e[k] = Σ_α u_e[α] * grad_phi[α, k]
    gradU_e = jnp.einsum('ea,eak->ek', u_e, grad_phi)   # (E, 3)

    # ----- Energy -----
    # E1 = -(mu0/2) * Σ_e V_e * |∇u_e|^2
    E1_e = -(MU0 / 2.0) * Ve * jnp.sum(gradU_e**2, axis=1)  # (E,)

    # For the load-like term, use midpoint rule:
    # dot_beta[e, β] = (Σ_α m_e[α]) · ∇φ_β
    sum_m_e = jnp.sum(m_e, axis=1)                          # (E, 3)
    dot_beta = jnp.einsum('ec,ebc->eb', sum_m_e, grad_phi)  # (E, 4)

    # E2 = mu0 * Σ_e Ms_e * V_e/4 * Σ_β u_e[β] * dot_beta[e,β]
    E2_e = MU0 * (Ms_e * Ve / 4.0) * jnp.einsum('eb,eb->e', u_e, dot_beta)  # (E,)

    S = jnp.sum(E1_e + E2_e)

    # ----- Gradient w.r.t u -----
    # Avoid forming the 4x4 K_e explicitly; contract directly:
    # term from -(mu0) * V_e * (K_e u)_β = -mu0 * V_e * Σ_α (∇φ_α·∇φ_β) u_α
    grad_u_term1 = -MU0 * jnp.einsum('eak,ebk,eb->ea', grad_phi, grad_phi, u_e) * Ve[:, None]  # (E,4)

    # term from + mu0 * Ms_e * V_e/4 * dot_beta
    grad_u_term2 = (MU0 * (Ms_e * Ve / 4.0))[:, None] * dot_beta                               # (E,4)

    grad_u_e = grad_u_term1 + grad_u_term2                                                     # (E,4)
    grad_u = jnp.zeros_like(u_nodes).at[conn].add(grad_u_e)                                    # (N,)

    # ----- Gradient w.r.t m -----
    # ∂s/∂m_e[γ,:] = mu0 * Ms_e * V_e/4 * ∇u_e  (same for all 4 local nodes)
    contrib_m_e = (MU0 * Ms_e * Ve / 4.0)[:, None, None] * gradU_e[:, None, :]                 # (E,4,3)
    grad_m = jnp.zeros_like(m_nodes).at[conn].add(contrib_m_e)                                 # (N,3)

    return S, grad_m, grad_u

__all__ = [
    "MU0",
    "magnetic_volume",
    "exchange_energy_and_grad",
    "uniaxial_anisotropy_energy_and_grad",
    "zeeman_energy_uniform_field_and_grad",
    "brown_energy_and_grad_from_m",
    "brown_energy_and_grad_from_scalar_potential",
]
