# optimize.py
"""
L-BFGS driver with configurable H0 seed (gamma / identity / diagonal / block-Jacobi).
CG-preconditioner code has been removed.

Exports:
- prepare_core_amg(...)
- prepare_core_amg_scalar(...)
- precompute_diag_tangent_from_geom(...)
- precompute_block_jacobi_3x3_from_geom(...)
- make_uniform_u_raw(...)
- minimize_energy_lbfgs(...)
"""
from __future__ import annotations
from typing import Optional, Tuple, Any, Callable
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
from jaxopt import LBFGS
from geom import TetGeom
from energies import (
    brown_energy_and_grad_from_m,
    brown_energy_and_grad_from_scalar_potential,  # <-- scalar functional
    exchange_energy_and_grad,
    uniaxial_anisotropy_energy_and_grad,
    zeeman_energy_uniform_field_and_grad,
)
from magnetostatics import (
    get_or_build_jax_amg_hierarchy,
    get_or_build_jax_amg_hierarchy_for_scalar,      # <-- scalar AMG
    _solve_A_jax_cg_compMG_core_jit,                # vector solve
    _solve_U_jax_cg_compMG_core_jit,                # <-- scalar solve
)

# =============================================================================
# AMG prep (host-side)
# =============================================================================
def prepare_core_amg(geom: TetGeom, *, amg: str = "sa", gauge: float = 0.0):
    hier = get_or_build_jax_amg_hierarchy(geom, amg=amg, gauge=gauge)
    A_t, P_t, R_t, Dinv_t = hier.A, hier.P, hier.R, hier.Dinv
    # Build SPD factor for the coarsest level (tiny system)
    A_c = A_t[-1].todense()
    A_c = 0.5 * (A_c + A_c.T)
    A_c = A_c + (1e-12) * jnp.eye(A_c.shape[0], dtype=A_c.dtype)
    L_c = jnp.linalg.cholesky(A_c)
    return A_t, P_t, R_t, Dinv_t, L_c


def prepare_core_amg_scalar(geom: TetGeom, *, amg: str = "sa"):
    """
    Scalar magnetostatics: build AMG on A = μ0*K (no gauge).
    """
    hier = get_or_build_jax_amg_hierarchy_for_scalar(geom, amg=amg)
    A_t, P_t, R_t, Dinv_t = hier.A, hier.P, hier.R, hier.Dinv
    # Coarsest SPD factor
    A_c = A_t[-1].todense()
    A_c = 0.5 * (A_c + A_c.T)
    A_c = A_c + (1e-12) * jnp.eye(A_c.shape[0], dtype=A_c.dtype)
    L_c = jnp.linalg.cholesky(A_c)
    return A_t, P_t, R_t, Dinv_t, L_c

# =============================================================================
# Diagonal / block-Jacobi SPD surrogates for H_ex + H_an
# =============================================================================
def precompute_diag_tangent_from_geom(
    geom: TetGeom,
    A_lookup_exchange: jnp.ndarray,  # (G,)
    K1_lookup: jnp.ndarray,          # (G,)
    *,
    E_ref: jnp.ndarray | None = None,
    mu: float = 0.0,
    eps: float = 1e-30,
) -> jnp.ndarray:
    """
    Per-node diagonal approximation for Hex+Han on the u-space tangent plane.
    Returns a positive (N,) diagonal, optionally scaled by 1/E_ref, with +mu damping.
    """
    conn = geom.conn
    grad_phi = geom.grad_phi
    volume = geom.volume
    mat_id = geom.mat_id
    N = (jnp.max(conn) + 1).astype(conn.dtype)
    g_ids = mat_id - 1
    A_e = A_lookup_exchange[g_ids]
    K1_e = K1_lookup[g_ids]

    # Exchange Jacobi
    grad_norm_sq = jnp.sum(grad_phi * grad_phi, axis=-1)  # (E,4)
    c_ex = (2.0 * A_e)[:, None] * volume[:, None] * grad_norm_sq  # (E,4)
    diag_ex = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        diag_ex = diag_ex.at[conn[:, l]].add(c_ex[:, l])

    # Anisotropy surrogate
    c_an = jnp.abs(K1_e) * (volume / 5.0)  # (E,)
    diag_an = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        diag_an = diag_an.at[conn[:, l]].add(c_an)

    diag = diag_ex + diag_an + jnp.asarray(mu, diag_ex.dtype)

    if E_ref is not None:
        one = jnp.array(1.0, dtype=diag.dtype)
        inv_ref = one / jnp.where(E_ref > 0.0, E_ref, one)
        diag = diag * inv_ref

    diag = jnp.maximum(diag, jnp.asarray(eps, diag.dtype))
    return diag


def precompute_block_jacobi_3x3_from_geom(
    geom: TetGeom,
    A_lookup_exchange: jnp.ndarray,  # (G,)
    K1_lookup: jnp.ndarray,          # (G,)
    k_easy_e: jnp.ndarray,           # (E,3)
    *,
    E_ref: jnp.ndarray | None = None,
    mu: float = 0.0,
    eps: float = 1e-30,
    return_inverse: bool = True,
) -> jnp.ndarray:
    """
    Per-node 3x3 SPD block-Jacobi for Hex+Han in the lab frame.
    M_i = (d_ex[i]) I3 + sum_{e∋i} 2*K1_e*(V_e/10)*(I - e_e e_e^T) + mu I3 (+ E_ref scaling)
    """
    conn = geom.conn
    grad_phi = geom.grad_phi
    Ve = geom.volume
    mat_id = geom.mat_id
    g_ids = mat_id - 1
    A_e = A_lookup_exchange[g_ids]
    K1_e = K1_lookup[g_ids]
    N = (jnp.max(conn) + 1).astype(conn.dtype)

    # Exchange isotropic diagonal per node
    grad_norm_sq = jnp.sum(grad_phi * grad_phi, axis=-1)  # (E,4)
    c_ex = (2.0 * A_e)[:, None] * Ve[:, None] * grad_norm_sq  # (E,4)
    d_ex = jnp.zeros((N,), dtype=grad_phi.dtype)
    for l in range(4):
        d_ex = d_ex.at[conn[:, l]].add(c_ex[:, l])

    # Anisotropy SPD per element
    I3 = jnp.eye(3, dtype=grad_phi.dtype)
    e_hat = k_easy_e  # (E,3)
    P_tan_e = I3[None, :, :] - jnp.einsum('ei,ej->eij', e_hat, e_hat)  # (E,3,3)
    coef_an = (2.0 * jnp.abs(K1_e)) * (Ve / 10.0)  # (E,)

    M = jnp.zeros((N, 3, 3), dtype=grad_phi.dtype)
    add_block = coef_an[:, None, None] * P_tan_e
    for l in range(4):
        M = M.at[conn[:, l]].add(add_block)

    M = M + d_ex[:, None, None] * I3[None, :, :]
    M = M + (jnp.asarray(mu, M.dtype) * I3)[None, :, :]

    if E_ref is not None:
        one = jnp.array(1.0, dtype=M.dtype)
        inv_ref = one / jnp.where(E_ref > 0.0, E_ref, one)
        M = M * inv_ref

    M = M + (jnp.asarray(eps, M.dtype) * I3)[None, :, :]

    if return_inverse:
        return jnp.linalg.inv(M)
    else:
        return M

# =============================================================================
# Utilities: pytrees and mapping u <-> m (chain rule)
# =============================================================================
def _tree_vdot(a, b) -> jnp.ndarray:
    la = jtu.tree_leaves(a)
    lb = jtu.tree_leaves(b)
    if not la:
        return jnp.array(0.0, dtype=jnp.float32)
    dtype = jnp.result_type(*([x.dtype for x in la] + [y.dtype for y in lb]))
    acc = jnp.array(0.0, dtype=dtype)
    for x, y in zip(la, lb):
        acc = acc + jnp.vdot(x, y).real
    return acc


def _u_raw_to_m_and_grad_u(
    u_raw: jnp.ndarray,
    grad_m_norm: jnp.ndarray,
    eps_norm: float = 1e-12
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    m = u / ||u|| ; ∇_u E = (I - m m^T) ∇_m E / ||u||
    """
    r = jnp.linalg.norm(u_raw, axis=1)
    r_safe = jnp.maximum(r, jnp.asarray(eps_norm, u_raw.dtype))
    m_nodes = u_raw / r_safe[:, None]
    gm_dot_m = jnp.sum(grad_m_norm * m_nodes, axis=1, keepdims=True)
    grad_u_raw = (grad_m_norm - gm_dot_m * m_nodes) / r_safe[:, None]
    return m_nodes, grad_u_raw

def _normalize_u(params: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """Row-wise normalize u to unit vectors; leaves near-zero rows at eps scale."""
    r = jnp.linalg.norm(params, axis=1, keepdims=True)
    r_safe = jnp.maximum(r, jnp.asarray(eps, params.dtype))
    return params / r_safe

def _apply_precond_tangent(
    params: jnp.ndarray,
    r: jnp.ndarray,
    *,
    precond_mode: str = "none",   # "none" | "diag" | "block_jacobi"
    diag_u=None,                  # (N,) for "diag"  or (N,3,3) Minv for "block_jacobi"
    eps: float = 1e-12,
) -> jnp.ndarray:
    """
    Project r onto the tangent at m = normalize(params), then apply the preconditioner P.

    Returns P * r_tan with:
      - "none": identity   (P = I)
      - "diag": P = diag^{-1} on the tangent (per-node scalar inverse)
      - "block_jacobi": P = Minv (N,3,3) on the tangent

    Shapes:
      params, r : (N, 3)
      diag_u    : (N,) for "diag", or (N,3,3) for "block_jacobi"
    """
    m = _normalize_u(params, eps=eps)                 # (N,3)
    r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m

    if precond_mode == "none" or diag_u is None:
        return r_tan
    elif precond_mode == "diag":
        diag = diag_u  # (N,)
        inv_diag = 1.0 / jnp.maximum(diag, jnp.asarray(1e-30, diag.dtype))
        return r_tan * inv_diag[:, None]
    elif precond_mode == "block_jacobi":
        Minv = diag_u  # (N,3,3)
        return jnp.einsum("nij,nj->ni", Minv, r_tan)
    else:
        raise ValueError(f"unknown precond_mode={precond_mode!r}")

def _curvilinear_step(params: jnp.ndarray,
                      d_tan: jnp.ndarray,
                      tau: jnp.ndarray) -> jnp.ndarray:
    """
    Curvilinear update on the sphere S^2 per node (Goldfarb–Wen–Yin).
    params : (N,3) current vectors (will be normalized internally)
    d_tan  : (N,3) tangent directions at params (dot per row ~ 0)
    tau    : scalar (broadcastable) step length
    Returns a new (N,3) array with ||row|| preserved.
    """
    U = _normalize_u(params)           # ensure unit vectors
    H = jnp.cross(U, d_tan)            # H = U × d  => U'(0)=d
    a, b, c = H[:, 0], H[:, 1], H[:, 2]
    u, v, w = U[:, 0], U[:, 1], U[:, 2]
    tau2 = tau * tau
    D = 4.0 + tau2 * (a*a + b*b + c*c)

    u_num = (4.0*u + 4.0*tau*b*w - 4.0*tau*c*v
             - tau2*b*b*u + tau2*a*a*u - tau2*c*c*u
             + 2.0*tau2*a*b*v + 2.0*tau2*a*c*w)
    v_num = (4.0*v + 4.0*tau*c*u - 4.0*tau*a*w
             + tau2*b*b*v - tau2*a*a*v - tau2*c*c*v
             + 2.0*tau2*c*b*w + 2.0*tau2*a*b*u)
    w_num = (4.0*w + 4.0*tau*a*v - 4.0*tau*b*u
             - tau2*b*b*w - tau2*a*a*w + tau2*c*c*w
             + 2.0*tau2*b*c*v + 2.0*tau2*a*c*u)

    return jnp.stack([u_num / D, v_num / D, w_num / D], axis=1)

def _make_Hcross(H: jnp.ndarray) -> jnp.ndarray:
    """
    Build the skew-symmetric cross-product matrices H_× for a batch of N vectors H.
    Input:
      H : (N, 3)
    Output:
      Hx: (N, 3, 3), where for each node i,
           Hx[i] = [[   0, -c,  b],
                    [   c,  0, -a],
                    [  -b,  a,  0]]
           with (a,b,c) = H[i]
    """
    a, b, c = H[:, 0], H[:, 1], H[:, 2]          # (N,)
    z = jnp.zeros_like(a)                         # (N,)
    # Build rows (N,3)
    row0 = jnp.stack([z,   -c,   b], axis=1)
    row1 = jnp.stack([c,    z,  -a], axis=1)
    row2 = jnp.stack([-b,   a,   z], axis=1)
    # Stack rows along axis=1 → (N,3,3)
    Hx = jnp.stack([row0, row1, row2], axis=1)
    return Hx


def _uprime_on_curve(params: jnp.ndarray,
                     d_tan: jnp.ndarray,
                     tau: jnp.ndarray) -> jnp.ndarray:
    """
    U'(tau) for the GWY curvilinear path.
    Uses Eq. (2.22): U'(tau) = 0.5 * Wminus^{-1} * (H_×) * (U + U(tau)),
    where Wminus = I - (tau/2) * H_× and H = U × d_tan.

    Shapes:
      params, d_tan : (N,3)
      tau           : scalar (broadcastable)
      return        : (N,3)
    """
    # Ensure unit vectors for the center of the curve
    U = _normalize_u(params)                      # (N,3)
    U_tau = _curvilinear_step(U, d_tan, tau)      # (N,3)

    # H = U × d (per node), and its cross-product matrix H_× (N,3,3)
    H = jnp.cross(U, d_tan)                       # (N,3)
    Hx = _make_Hcross(H)                          # (N,3,3)  <-- fixed, no transpose needed

    # Batched 3x3 system: Wminus * x = H_× (U + U_tau)
    I3 = jnp.eye(3, dtype=U.dtype)                # (3,3)
    # Broadcast I3 across batch; tau is scalar → broadcast ok
    Wminus = I3[None, :, :] - 0.5 * tau * Hx      # (N,3,3)
    rhs = jnp.einsum("nij,nj->ni", Hx, (U + U_tau))  # (N,3)

    # Solve per node
    def _solve(A, b): return jnp.linalg.solve(A, b)
    sol = jax.vmap(_solve, in_axes=(0, 0))(Wminus, rhs)  # (N,3)
    return 0.5 * sol

def _curvilinear_wolfe_linesearch(
    fun_value_and_grad,
    params: jnp.ndarray,
    f0: jnp.ndarray,
    g0: jnp.ndarray,
    aux_state: Any,              # either (aux_flat, parts) or aux_flat
    d_tan: jnp.ndarray,
    tau_init: float,
    c1: float = 1e-4,
    c2: float = 0.9,
    tau_max: float = 1.0,
    grow: float = 2.0,
    max_bracket: int = 20,
    max_zoom: int = 30,
):
    """
    Weak Wolfe LS along U(tau) (GWY curve):
      - Bracket phase (expand by 'grow' when needed),
      - Zoom phase (bisection; interpolation can be added later).
    Returns:
      (tau_accepted, new_params, new_value, new_grad,
       aux_flat_new, parts_new,
       ls_iter_total, failed, num_fun_evals)
    """
    tau0 = jnp.asarray(tau_init, dtype=params.dtype)
    c1 = jnp.asarray(c1, dtype=params.dtype)
    c2 = jnp.asarray(c2, dtype=params.dtype)
    tau_max = jnp.asarray(tau_max, dtype=params.dtype)
    grow = jnp.asarray(grow, dtype=params.dtype)

    # Slope at zero
    dphi0 = _tree_vdot(g0, d_tan)  # scalar (negative for descent)
    # Normalize aux state pieces
    if isinstance(aux_state, tuple) and len(aux_state) > 1:
        aux_flat0, parts0 = aux_state
    else:
        aux_flat0, parts0 = aux_state, jnp.zeros((5,), dtype=f0.dtype)

    # Common evaluator at tau
    def _eval_tau(tau):
        params_tau = _curvilinear_step(params, d_tan, tau)
        (f_tau, (aux_flat, parts)), g_tau = fun_value_and_grad(params_tau, aux_flat0)
        # Directional derivative phi'(tau) = <g_tau, U'(tau)>
        Uprime_tau = _uprime_on_curve(params, d_tan, tau)
        dphi_tau = _tree_vdot(g_tau, Uprime_tau)
        return params_tau, f_tau, g_tau, aux_flat, parts, dphi_tau

    # ---- Bracket phase ----
    # Carry: tau, it, has_high, t_low, f_low, d_low, t_high, f_high, d_high, fe, done, acc_tau, acc_pack...
    init_carry = (
        tau0, jnp.asarray(0, jnp.int32),
        jnp.asarray(False, jnp.bool_),
        jnp.asarray(0.0, params.dtype), f0, dphi0,       # t_low, f_low, d_low
        jnp.asarray(0.0, params.dtype), f0, dphi0,       # t_high, f_high, d_high (unused until has_high)
        jnp.asarray(0, jnp.int32),                       # fe count
        jnp.asarray(False, jnp.bool_),                   # done
        jnp.asarray(0.0, params.dtype),                  # acc_tau
        params, f0, g0, aux_flat0, parts0                # accepted pack (init as base)
    )

    def bracket_cond(c):
        tau, it, has_high, tL, fL, dL, tH, fH, dH, fe, done, acc_tau, p_acc, f_acc, g_acc, ax_acc, pr_acc = c
        return jnp.logical_and(it < max_bracket, jnp.logical_not(done))

    def bracket_body(c):
        tau, it, has_high, tL, fL, dL, tH, fH, dH, fe, done, acc_tau, p_acc, f_acc, g_acc, ax_acc, pr_acc = c

        p_t, f_t, g_t, ax_t, pr_t, d_t = _eval_tau(tau)
        fe = fe + 1

        armijo_fail = f_t > (f0 + c1 * tau * dphi0)
        nonmonotone = jnp.logical_and(it > 0, f_t >= fL)

        # Case 1: Armijo fail OR f(t) >= f(t_low)  -> bracket found [t_low, t]
        set_high = jnp.logical_or(armijo_fail, nonmonotone)

        # Accept if curvature holds (weak Wolfe): dphi(t) >= c2 * dphi(0)
        wolfe_ok = d_t >= (c2 * dphi0)

        # Decide next:
        # If set_high -> enter zoom (done=True)
        # Else if wolfe_ok -> accept (done=True; acc = current)
        # Else if dphi(t) >= 0 -> set_high and zoom (done=True)
        become_done = jnp.logical_or(set_high, jnp.logical_or(wolfe_ok, (d_t >= 0.0)))

        # Update bracket data
        new_has_high = jnp.logical_or(has_high, set_high | (d_t >= 0.0))
        new_tH = jax.lax.select(set_high | (d_t >= 0.0), tau, tH)
        new_fH = jax.lax.select(set_high | (d_t >= 0.0), f_t, fH)
        new_dH = jax.lax.select(set_high | (d_t >= 0.0), d_t, dH)

        # Update accepted package if wolfe_ok
        acc_tau2 = jax.lax.select(wolfe_ok, tau, acc_tau)
        p_acc2   = jax.lax.select(wolfe_ok, p_t, p_acc)
        f_acc2   = jax.lax.select(wolfe_ok, f_t, f_acc)
        g_acc2   = jax.lax.select(wolfe_ok, g_t, g_acc)
        ax_acc2  = jax.lax.select(wolfe_ok, ax_t, ax_acc)
        pr_acc2  = jax.lax.select(wolfe_ok, pr_t, pr_acc)

        # If not done and not setting high, advance t_low = t and grow tau
        new_tL = jax.lax.select(set_high | wolfe_ok | (d_t >= 0.0), tL, tau)
        new_fL = jax.lax.select(set_high | wolfe_ok | (d_t >= 0.0), fL, f_t)
        new_dL = jax.lax.select(set_high | wolfe_ok | (d_t >= 0.0), dL, d_t)

        tau_next = jax.lax.select(
            become_done, tau, jnp.minimum(tau * grow, tau_max)
        )

        return (tau_next, it + 1,
                new_has_high,
                new_tL, new_fL, new_dL,
                new_tH, new_fH, new_dH,
                fe, become_done,
                acc_tau2, p_acc2, f_acc2, g_acc2, ax_acc2, pr_acc2)

    tau_b, it_b, has_high_b, tL_b, fL_b, dL_b, tH_b, fH_b, dH_b, fe_b, done_b, acc_tau_b, p_acc_b, f_acc_b, g_acc_b, ax_acc_b, pr_acc_b = \
        jax.lax.while_loop(bracket_cond, bracket_body, init_carry)

    # If accepted during bracket, return it; else go to zoom if we have a high; else fallback to last bracket tau
    accepted = jnp.logical_and(done_b, acc_tau_b > 0.0)
    have_bracket = jnp.logical_and(done_b, has_high_b)

    def _return_accepted(_):
        return (acc_tau_b, p_acc_b, f_acc_b, g_acc_b, ax_acc_b, pr_acc_b,
                it_b, jnp.asarray(False, jnp.bool_), fe_b)

    def _fallback_no_bracket(_):
        # Use last evaluated tau_b (best we have)
        p_t, f_t, g_t, ax_t, pr_t, _ = _eval_tau(tau_b)
        return (tau_b, p_t, f_t, g_t, ax_t, pr_t,
                it_b, jnp.asarray(True, jnp.bool_), fe_b + 1)

    # ---- Zoom phase (bisection) ----
    def _run_zoom(_):
        # zoom carry: tL, fL, dL, tH, fH, dH, it, fe, acc_flag, acc_pack...
        carry_z = (tL_b, fL_b, dL_b, tH_b, fH_b, dH_b, it_b, fe_b,
                   jnp.asarray(False, jnp.bool_),
                   acc_tau_b, p_acc_b, f_acc_b, g_acc_b, ax_acc_b, pr_acc_b)

        def zoom_cond(z):
            tL, fL, dL, tH, fH, dH, it, fe, acc, acc_tau, p_acc, f_acc, g_acc, ax_acc, pr_acc = z
            return jnp.logical_and(it < (it_b + max_zoom), jnp.logical_not(acc))

        def zoom_body(z):
            tL, fL, dL, tH, fH, dH, it, fe, acc, acc_tau, p_acc, f_acc, g_acc, ax_acc, pr_acc = z
            tau = 0.5 * (tL + tH)
            p_t, f_t, g_t, ax_t, pr_t, d_t = _eval_tau(tau)
            fe = fe + 1

            cond1 = f_t > (f0 + c1 * tau * dphi0)
            cond2 = f_t >= fL

            case_left = jnp.logical_or(cond1, cond2)  # move high to tau
            new_tH = jax.lax.select(case_left, tau, tH)
            new_fH = jax.lax.select(case_left, f_t, fH)
            new_dH = jax.lax.select(case_left, d_t, dH)

            # If not case_left, test curvature
            wolfe_ok = d_t >= (c2 * dphi0)

            # If wolfe ok, accept
            acc2 = jnp.logical_or(acc, wolfe_ok)
            acc_tau2 = jax.lax.select(wolfe_ok, tau, acc_tau)
            p_acc2   = jax.lax.select(wolfe_ok, p_t, p_acc)
            f_acc2   = jax.lax.select(wolfe_ok, f_t, f_acc)
            g_acc2   = jax.lax.select(wolfe_ok, g_t, g_acc)
            ax_acc2  = jax.lax.select(wolfe_ok, ax_t, ax_acc)
            pr_acc2  = jax.lax.select(wolfe_ok, pr_t, pr_acc)

            # Otherwise, decide side based on derivative sign
            # If d_t * (tH - tL) >= 0 -> set tH = tL
            prod = d_t * (new_tH - tL)
            tH_fix = jax.lax.select(prod >= 0.0, tL, new_tH)
            fH_fix = jax.lax.select(prod >= 0.0, fL, new_fH)
            dH_fix = jax.lax.select(prod >= 0.0, dL, new_dH)

            # Move low to tau when not case_left and not accepted
            move_low = jnp.logical_and(jnp.logical_not(case_left), jnp.logical_not(wolfe_ok))
            new_tL = jax.lax.select(move_low, tau, tL)
            new_fL = jax.lax.select(move_low, f_t, fL)
            new_dL = jax.lax.select(move_low, d_t, dL)

            return (new_tL, new_fL, new_dL, tH_fix, fH_fix, dH_fix,
                    it + 1, fe, acc2, acc_tau2, p_acc2, f_acc2, g_acc2, ax_acc2, pr_acc2)

        tL2, fL2, dL2, tH2, fH2, dH2, it2, fe2, acc2, acc_tau2, p_acc2, f_acc2, g_acc2, ax_acc2, pr_acc2 = \
            jax.lax.while_loop(zoom_cond, zoom_body, carry_z)

        # If accepted, return; else fallback to mid
        final_tau = jax.lax.select(acc2, acc_tau2, 0.5 * (tL2 + tH2))
        p_t, f_t, g_t, ax_t, pr_t, _ = _eval_tau(final_tau)
        fe_out = fe2 + jnp.asarray(jnp.logical_not(acc2), jnp.int32)
        failed = jnp.logical_not(acc2)
        return (final_tau, p_t, f_t, g_t, ax_t, pr_t, it2, failed, fe_out)

    # Choose among accepted/bracketed/none
    return jax.lax.cond(
        accepted, _return_accepted,
        lambda _: jax.lax.cond(have_bracket, _run_zoom, _fallback_no_bracket, operand=None),
        operand=None
    )

def _curvilinear_backtracking_armijo(
    fun_value_and_grad,
    params: jnp.ndarray,
    f_k: jnp.ndarray,
    g_k: jnp.ndarray,
    aux_state: Any,             # either A0_flat (array) or (aux_flat, parts)
    d_tan: jnp.ndarray,
    tau_init: float,
    c1: float = 1e-4,
    decrease_factor: float = 0.5,
    maxiter: int = 60,
):
    """
    JAX-pure Armijo backtracking along U(tau) = _curvilinear_step(params, d_tan, tau).
    Returns:
      (tau_used, new_params, new_value, new_grad,
       aux_flat_new, parts_new, iters, failed, num_fun_evals)
    """
    tau0 = jnp.asarray(tau_init, dtype=params.dtype)
    c1 = jnp.asarray(c1, dtype=params.dtype)
    dec = jnp.asarray(decrease_factor, dtype=params.dtype)

    # Slope at zero for general tangent direction
    phi_prime0 = _tree_vdot(g_k, d_tan)

    # Pull an initial 'parts' shape from aux_state when available
    if isinstance(aux_state, tuple) and len(aux_state) > 1:
        parts0 = aux_state[1]
        aux_flat0 = aux_state[0]
    else:
        aux_flat0 = aux_state
        parts0 = jnp.zeros((5,), dtype=f_k.dtype)  # fallback

    def eval_at_tau(tau):
        trial_params = _curvilinear_step(params, d_tan, tau)
        (trial_val, (aux_flat_new, parts_new)), trial_grad = fun_value_and_grad(trial_params, aux_flat0)
        ok = trial_val <= f_k + c1 * tau * phi_prime0
        return trial_val, trial_grad, aux_flat_new, parts_new, trial_params, ok

    # Carry: tau, it, done, tau_acc, p_best, f_best, g_best, ax_best, pr_best, nfe
    carry0 = (
        tau0, jnp.asarray(0, jnp.int32), jnp.asarray(False, jnp.bool_),
        jnp.asarray(0.0, params.dtype),
        params, f_k, g_k,
        aux_flat0, parts0, jnp.asarray(0, jnp.int32)
    )

    def cond_fun(c):
        tau, it, done, *_ = c
        return jnp.logical_and(it < maxiter, jnp.logical_not(done))

    def body_fun(c):
        tau, it, done, tau_acc, p_best, f_best, g_best, ax_best, pr_best, nfe = c
        tv, tg, ax, pr, tp, ok = eval_at_tau(tau)

        p_new  = jax.lax.select(ok, tp, p_best)
        f_new  = jax.lax.select(ok, tv, f_best)
        g_new  = jax.lax.select(ok, tg, g_best)
        ax_new = jax.lax.select(ok, ax, ax_best)
        pr_new = jax.lax.select(ok, pr, pr_best)
        tau_acc_new = jax.lax.select(ok, tau, tau_acc)
        done_new = jnp.logical_or(done, ok)

        return (tau * dec, it + 1, done_new, tau_acc_new,
                p_new, f_new, g_new, ax_new, pr_new, nfe + 1)

    tau_next, it_out, done_out, tau_acc, p_out, f_out, g_out, ax_out, pr_out, nfe_out = \
        jax.lax.while_loop(cond_fun, body_fun, carry0)

    # Fallback: evaluate once at last tried tau
    last_tau = tau_next / dec
    tv, tg, ax, pr, tp, ok = eval_at_tau(last_tau)

    final_tau = jax.lax.select(done_out, tau_acc, last_tau)
    final_p   = jax.lax.select(done_out, p_out, tp)
    final_f   = jax.lax.select(done_out, f_out, tv)
    final_g   = jax.lax.select(done_out, g_out, tg)
    final_ax  = jax.lax.select(done_out, ax_out, ax)
    final_pr  = jax.lax.select(done_out, pr_out, pr)
    final_nfe = jax.lax.select(done_out, nfe_out, nfe_out + 1)
    failed    = jnp.logical_not(done_out)

    return (final_tau, final_p, final_f, final_g, final_ax, final_pr,
            it_out, failed, final_nfe)

def _curvilinear_modified_armijo(
    fun_value_and_grad,
    params: jnp.ndarray,
    f_k: jnp.ndarray,
    g_k: jnp.ndarray,
    aux_state: Any,               # either aux_flat or (aux_flat, parts)
    d_tan: jnp.ndarray,           # tangent direction at 'params'
    tau_init: float,
    *,
    eta1: float = 0.10,           # sufficient decrease target: D(τ) ≥ η1
    eta2: float = 0.10,           # not-too-small target: 1 - D(τ) ≥ η2
    C: float = 2.0,               # growth factor > 1
    shrink: float = 0.5,          # shrink factor in (0,1)  [renamed from c]
    max_enlarge: int = 20,
    max_reduce: int = 60,
    tau_max: float = 1.0,         # clamp to a reasonable bound (e.g., your max stepsize)
    eps_denom: float = 1e-16
):
    """
    Modified Armijo (Bartholomew-Biggs, Chap. 8) along the GWY curvilinear path.
    Returns (9-tuple), same shape/order as _curvilinear_backtracking_armijo:
        (tau_used, new_params, new_value, new_grad,
         aux_flat_new, parts_new, ls_iter_num, failed, num_fun_evals)
    """

    # ----------------------------
    # Setup & guards
    # ----------------------------
    tau0 = jnp.asarray(tau_init, dtype=params.dtype)
    C = jnp.asarray(C, dtype=params.dtype)
    shrink = jnp.asarray(shrink, dtype=params.dtype)   # <— use 'shrink' name
    eta1 = jnp.asarray(eta1, dtype=params.dtype)
    eta2 = jnp.asarray(eta2, dtype=params.dtype)
    tau_max = jnp.asarray(tau_max, dtype=params.dtype)
    eps = jnp.asarray(eps_denom, dtype=params.dtype)

    # aux_state unpacking: keep a 'parts' tensor for shape consistency
    if isinstance(aux_state, tuple) and len(aux_state) > 1:
        aux_flat0, parts0 = aux_state
    else:
        aux_flat0 = aux_state
        parts0 = jnp.zeros((5,), dtype=f_k.dtype)  # harmless fallback

    # φ'(0) = <g_k, U'(0)>, with our path U'(0) = d_tan
    phi_prime0 = _tree_vdot(g_k, d_tan)

    # If not descent (phi_prime0 >= 0), flip direction to be safe
    def _flip_dir(_):
        return jtu.tree_map(lambda x: -x, d_tan), -phi_prime0
    def _keep_dir(_):
        return d_tan, phi_prime0
    d_dir, phi0 = jax.lax.cond(phi_prime0 >= 0.0, _flip_dir, _keep_dir, operand=None)

    # Common evaluator at τ: computes D(τ) and returns trial pack
    # D(τ) = (f(τ) - f0) / (τ * φ'(0))
    def _eval_at_tau(tau):
        trial_params = _curvilinear_step(params, d_dir, tau)
        (val, (aux_flat_new, parts_new)), grad = fun_value_and_grad(trial_params, aux_flat0)
        denom = tau * phi0
        # Guard against tiny or zero denominator; keep sign consistent to avoid overflow
        denom = jnp.where(jnp.abs(denom) > eps, denom,
                          jnp.sign(jnp.where(denom == 0.0, phi0, denom)) * eps)
        D = (val - f_k) / denom
        return trial_params, val, grad, aux_flat_new, parts_new, D

    # Initial evaluation
    trial, val, grad, aux_flat, parts, D = _eval_at_tau(tau0)
    fe = jnp.asarray(1, jnp.int32)   # function evals used
    it = jnp.asarray(0, jnp.int32)   # line-search inner iterations

    # ----------------------------
    # Phase 1: Enlargement (ensure 1 - D >= η2)
    # ----------------------------
    # carry: τ, τ_min, it, fe, trial, val, grad, aux_flat, parts, D
    carry_e = (tau0, jnp.asarray(0.0, params.dtype), it, fe,
               trial, val, grad, aux_flat, parts, D)

    def cond_enlarge(carry):
        tau, tau_min, it, fe, trial, val, grad, auxf, prt, D = carry
        return jnp.logical_and(it < max_enlarge, (1.0 - D) < eta2)

    def body_enlarge(carry):
        tau, tau_min, it, fe, trial, val, grad, auxf, prt, D = carry
        denom = jnp.maximum(1.0 - D, eps)           # avoid division by zero
        tau_interp = 0.5 * tau / denom              # aim at D ≈ 0.5
        # If D >= 1 (nonconvex local shape), interpolation would go negative; use growth
        interp_ok = (1.0 - D) > 0.0
        tau_next = jax.lax.select(interp_ok, jnp.minimum(C * tau, tau_interp), C * tau)
        tau_next = jnp.minimum(tau_next, tau_max)

        trial2, val2, grad2, auxf2, prt2, D2 = _eval_at_tau(tau_next)
        # Update τ_min to last τ (book algorithm keeps a lower bound)
        return (tau_next, tau, it + 1, fe + 1, trial2, val2, grad2, auxf2, prt2, D2)

    tau_e, tau_min_e, it_e, fe_e, trial_e, val_e, grad_e, auxf_e, prt_e, D_e = \
        jax.lax.while_loop(cond_enlarge, body_enlarge, carry_e)

    # ----------------------------
    # Phase 2: Reduction (ensure D >= η1)
    # ----------------------------
    carry_r = (tau_e, tau_min_e, it_e, fe_e, trial_e, val_e, grad_e, auxf_e, prt_e, D_e)

    def cond_reduce(carry):
        tau, tau_min, it, fe, trial, val, grad, auxf, prt, D = carry
        return jnp.logical_and(it < (it_e + max_reduce), D < eta1)

    def body_reduce(carry):
        tau, tau_min, it, fe, trial, val, grad, auxf, prt, D = carry
        # Shrink toward τ_min, but also try to jump to D ≈ 0.5
        denom = jnp.maximum(1.0 - D, eps)           # avoid division by zero
        tau_interp = 0.5 * tau / denom
        tau_shrink = tau_min + shrink * (tau - tau_min)  # <— use 'shrink' scalar
        tau_next = jnp.maximum(jnp.maximum(tau_interp, tau_shrink),
                               jnp.asarray(1e-16, tau.dtype))
        tau_next = jnp.minimum(tau_next, tau_max)

        trial2, val2, grad2, auxf2, prt2, D2 = _eval_at_tau(tau_next)
        return (tau_next, tau_min, it + 1, fe + 1, trial2, val2, grad2, auxf2, prt2, D2)

    tau_r, tau_min_r, it_r, fe_r, trial_r, val_r, grad_r, auxf_r, prt_r, D_r = \
        jax.lax.while_loop(cond_reduce, body_reduce, carry_r)

    # ----------------------------
    # Final pack: success/failure and totals
    # ----------------------------
    success = D_r >= eta1
    failed = jnp.logical_not(success)

    tau_used = tau_r
    new_params = trial_r
    new_value = val_r
    new_grad = grad_r
    aux_flat_new = auxf_r
    parts_new = prt_r

    # Total LS iters = enlarge iters + reduce iters
    ls_iter_num = it_r   # 'it' was accumulated across both phases
    num_fun_evals = fe_r # total function+grad evaluations used by LS

    return (
        tau_used,          # (1)
        new_params,        # (2)
        new_value,         # (3)
        new_grad,          # (4)
        aux_flat_new,      # (5)
        parts_new,         # (6)
        ls_iter_num,       # (7)
        failed,            # (8)
        num_fun_evals      # (9)
    )


# =============================================================================
# Analytic value+grad with aux (A or U warm-start) — normalized by E_ref
# =============================================================================
def make_valgrad_with_aux(ms_mode: str):
    def valgrad(
        u_raw: jnp.ndarray,
        aux0_flat: Optional[jnp.ndarray],
        # AMG tuples & factor
        A_t, P_t, R_t, Dinv_t, L_c,
        # Geometry & materials
        conn: jnp.ndarray, grad_phi: jnp.ndarray, volume: jnp.ndarray, mat_id: jnp.ndarray,
        Ms_lookup: jnp.ndarray,
        A_lookup_exchange: jnp.ndarray,
        K1_lookup: jnp.ndarray,
        k_easy_e: jnp.ndarray,
        # Field & normalization
        H_ext: jnp.ndarray,
        E_ref: jnp.ndarray,
        # Solver params (static for specialization)
        tol: float, maxiter: int, nu_pre: int, nu_post: int, omega: float,
        coarse_iters: int, coarse_omega: float,
        # gauge for A-operator (ignored if ms_mode=='U')
        gauge: float,
        eps_norm: float = 1e-12,
    ):

        if isinstance(aux0_flat, (tuple, list)):
            aux0_warm = aux0_flat[0]  # take the flat A/U field only
        else:
            aux0_warm = aux0_flat

        # u -> m
        m_nodes, _ = _u_raw_to_m_and_grad_u(u_raw, jnp.zeros_like(u_raw), eps_norm)
        N = m_nodes.shape[0]
        if ms_mode == "A":
            x0_flat = jnp.zeros((3 * N,), dtype=jnp.float64) if aux0_warm is None else aux0_warm
        else:
            x0_flat = jnp.zeros((N,), dtype=jnp.float64) if aux0_warm is None else aux0_warm

        geom = TetGeom(conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
                       volume_scalefactor=jnp.asarray(1.0, dtype=jnp.float64))

        if ms_mode == "A":
            # Solve for vector potential A
            A_sol, *_ = _solve_A_jax_cg_compMG_core_jit(
                A_t, P_t, R_t, Dinv_t, L_c,
                x0_flat, m_nodes, geom, Ms_lookup,
                tol, maxiter, gauge=gauge,
                nu_pre=nu_pre, nu_post=nu_post, omega=omega,
                coarse_iters=coarse_iters, coarse_omega=coarse_omega
            )
            A_sol = jax.lax.stop_gradient(A_sol)
            aux_flat = A_sol.reshape(-1)
            S, grad_m_S, _ = brown_energy_and_grad_from_m(m_nodes, A_sol, geom, Ms_lookup)
        else:
            # Solve for scalar potential U (no gauge)
            U_sol, *_ = _solve_U_jax_cg_compMG_core_jit(
                A_t, P_t, R_t, Dinv_t, L_c,
                x0_flat, m_nodes, geom, Ms_lookup,
                tol, maxiter,
                nu_pre=nu_pre, nu_post=nu_post, omega=omega,
                coarse_iters=coarse_iters, coarse_omega=coarse_omega,
            )
            U_sol = jax.lax.stop_gradient(U_sol)
            aux_flat = U_sol.reshape(-1)
            S, grad_m_S, _ = brown_energy_and_grad_from_scalar_potential(m_nodes, U_sol, geom, Ms_lookup)

        E_ex, grad_m_ex = exchange_energy_and_grad(m_nodes, geom, A_lookup=A_lookup_exchange)
        E_an, grad_m_an = uniaxial_anisotropy_energy_and_grad(m_nodes, geom, K1_lookup, k_easy_e)
        E_z, grad_m_z = zeeman_energy_uniform_field_and_grad(m_nodes, geom, Ms_lookup, H_ext)

        E_total = S + E_ex + E_an + E_z
        grad_m = grad_m_S + grad_m_ex + grad_m_an + grad_m_z
        one = jnp.array(1.0, dtype=E_ref.dtype)
        E_ref_safe = jnp.where(E_ref > 0.0, E_ref, one)
        inv_ref = one / E_ref_safe
        E_total_norm = E_total * inv_ref

        _, grad_u_raw = _u_raw_to_m_and_grad_u(u_raw, grad_m, eps_norm)
        grad_u_norm = grad_u_raw * inv_ref
                
        parts_norm = jnp.stack(
            [S, E_ex, E_an, E_z, E_total],  # keep total too for convenience
            dtype=E_total.dtype
        ) * inv_ref
        
        return (E_total_norm, (aux_flat, parts_norm)), grad_u_norm

    return jax.jit(
        valgrad,
        static_argnames=("tol", "maxiter", "nu_pre", "nu_post", "coarse_iters")
    )

# Precompile two specializations so mode selection is fast at runtime
_VALGRAD_A = make_valgrad_with_aux("A")
_VALGRAD_U = make_valgrad_with_aux("U")

# =============================================================================
# Two-loop with custom H0 (no CG branch)
# =============================================================================
def _compute_gamma_scalar(s_history, y_history, last: int):
    y_last = jtu.tree_map(lambda H: H[last], y_history)
    s_last = jtu.tree_map(lambda H: H[last], s_history)
    num = _tree_vdot(y_last, s_last)
    denom = _tree_vdot(y_last, y_last)
    return jnp.where(denom > 0.0, num / denom, 1.0)


def _inv_hessian_product_H0(
    pytree: Any,
    s_history: Any, y_history: Any, rho_history: jnp.ndarray,
    start: int,
    H0_matvec: Callable[[Any], Any],
):
    """Standard two-loop with custom middle multiply r = H0 r."""
    m = rho_history.shape[0]
    idx = (start + jnp.arange(m)) % m

    def right_body(carry, i):
        r = carry
        si = jtu.tree_map(lambda h: h[i, ...], s_history)
        yi = jtu.tree_map(lambda h: h[i, ...], y_history)
        alpha = rho_history[i] * _tree_vdot(si, r)
        r = jtu.tree_map(lambda rL, yL: rL - alpha * yL, r, yi)
        return r, alpha

    r, alphas = jax.lax.scan(right_body, pytree, idx, reverse=True)
    r = H0_matvec(r)

    def left_body(carry, data):
        r = carry
        i, alpha = data
        si = jtu.tree_map(lambda h: h[i, ...], s_history)
        yi = jtu.tree_map(lambda h: h[i, ...], y_history)
        beta = rho_history[i] * _tree_vdot(yi, r)
        r = jtu.tree_map(lambda rL, sL: rL + (alpha - beta) * sL, r, si)
        return r, beta

    r, _ = jax.lax.scan(left_body, r, (idx, alphas))
    return r

# =============================================================================
# Make JAXopt callable
# =============================================================================
def make_fun_for_jaxopt(
    *,
    A_t, P_t, R_t, Dinv_t, L_c,
    conn, grad_phi, volume, mat_id,
    Ms_lookup, A_lookup_exchange, K1_lookup, k_easy_e,
    H_ext, E_ref,
    tol=1e-3, maxiter=500, nu_pre=2, nu_post=2, omega=0.7,
    coarse_iters=8, coarse_omega=0.7,
    gauge: float = 0.0,  # for A only
    ms_mode: str = "A",
    eps_norm=1e-12,
):
    def fun_value_and_grad(params, A0_flat):
        VALGRAD = _VALGRAD_A if ms_mode == "A" else _VALGRAD_U
        return VALGRAD(
            params, A0_flat,
            A_t, P_t, R_t, Dinv_t, L_c,
            conn, grad_phi, volume, mat_id,
            Ms_lookup, A_lookup_exchange, K1_lookup, k_easy_e,
            H_ext, E_ref,
            tol, maxiter, nu_pre, nu_post, omega, coarse_iters, coarse_omega,
            gauge,  # ignored in scalar mode
            eps_norm,
        )
    return fun_value_and_grad

def _eta_schedule_piecewise(iter_num: jnp.ndarray, dtype=jnp.float64):
    """
    Piecewise schedule for (eta1, eta2), returned with the same dtype as params.

    Iteration   0–9  : 0.45
    Iteration  10–49 : 0.30
    Iteration >= 50  : 0.10
    """
    it = jnp.asarray(iter_num, dtype=jnp.int32)
    eta = jnp.where(it < 10,
                    jnp.asarray(0.45, dtype=dtype),
                    jnp.where(it < 50,
                              jnp.asarray(0.30, dtype=dtype),
                              jnp.asarray(0.10, dtype=dtype)))
    return eta, eta  # (eta1, eta2)

def _eta_schedule_smooth(iter_num: jnp.ndarray, dtype=jnp.float64):
    """
    Smooth schedule in [0.10, 0.45]:
      eta(it) = 0.10 + (0.45 - 0.10) * exp(-it / 20)
    """
    it = jnp.asarray(iter_num, dtype=jnp.float64)
    eta = jnp.asarray(0.10, dtype=dtype) + \
          jnp.asarray(0.35, dtype=dtype) * jnp.exp(-it / jnp.asarray(20.0, dtype=dtype))
    # Clamp for numerical safety
    eta = jnp.clip(eta, jnp.asarray(0.10, dtype=dtype), jnp.asarray(0.45, dtype=dtype))
    return eta, eta

# =============================================================================
# Two-loop L-BFGS driver (no CG branch)
# =============================================================================

def run_jaxopt_lbfgs_twoloop(
    init_params,
    init_aux_flat,
    *,
    fun_value_and_grad,
    history_size: int,
    max_iter: int,
    grad_tol: float,
    H0_mode: str = "gamma",
    ls_init_mode: str = "increase",
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    ls_increase_factor: float = 1.5,
    debug: bool = False,
    ls_kind: str = "default",  
    ls_c1: float = 0.3,
    ls_c2: float = 0.7,
    ls_decrease: float = 0.5,
    ls_increase: float = 2.0,
    diag_u=None,
):
    solver = LBFGS(
        fun=fun_value_and_grad,
        value_and_grad=True,
        has_aux=True,
        maxiter=max_iter,
        tol=grad_tol,
        history_size=history_size,
        verbose=True,
        implicit_diff=False,
        max_stepsize=ls_max_stepsize,
        maxls=60,
    )

    params0 = init_params
    state0 = solver.init_state(params0, init_aux_flat)
    f_init = state0.value
    carry0 = (params0, state0, params0, f_init)

    TauF = jnp.asarray(grad_tol, dtype=params0.dtype)
    TauF_sqrt = jnp.sqrt(TauF)
    TauF_cuberoot = TauF ** (1.0 / 3.0)
    epsM = jnp.asarray(jnp.finfo(params0.dtype).eps, dtype=params0.dtype)

    def cond_fun(carry):
        params, state, x_prev, f_prev = carry
        f_k = state.value
        g_k = state.grad
        it = state.iter_num
        x_norm = jnp.linalg.norm(params, ord=jnp.inf)
        step_norm = jnp.linalg.norm(params - x_prev, ord=jnp.inf)
        g_norm = jnp.linalg.norm(g_k, ord=jnp.inf)
        eta_hat = TauF * (1.0 + jnp.abs(f_k))
        eps_A = epsM * (1.0 + jnp.abs(f_k))
        fun_ok = jax.lax.cond(it > 0, lambda _: jnp.abs(f_prev - f_k) <= eta_hat, lambda _: False, operand=None)
        step_ok = jax.lax.cond(it > 0, lambda _: step_norm <= (TauF_sqrt * (1.0 + x_norm)), lambda _: False, operand=None)
        grad_ok = g_norm <= (TauF_cuberoot * (1.0 + jnp.abs(f_k)))
        grad_abs_ok = g_norm < eps_A

        '''
        if debug:
            jax.debug.print(
                "[LBFGS it:{:03d}] f {:+.9e}  |g|_inf {:.3e}  Δx_inf {:.3e}  U:{} {} {} {}",
                it, f_k, g_norm, step_norm,
                fun_ok.astype(jnp.int32), step_ok.astype(jnp.int32),
                grad_ok.astype(jnp.int32), grad_abs_ok.astype(jnp.int32)
            )
        '''

        success = (fun_ok & step_ok & grad_ok) | grad_abs_ok
        return jnp.logical_and(~success, it < max_iter)

    def body_fun(carry):
        params, state, x_prev, f_prev = carry

        # Build H0 operator
        if H0_mode == "gamma":
            gamma = state.gamma
            H0_matvec = lambda r, g=gamma: jtu.tree_map(lambda x: g * x, r)
        elif H0_mode == "identity":
            H0_matvec = lambda r: r
        elif H0_mode == "diag":
            diag = diag_u
            def H0_matvec(r):
                u = params
                m = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), jnp.asarray(1e-12, u.dtype))
                r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m
                return r_tan * (1.0 / jnp.maximum(diag, jnp.asarray(1e-30, diag.dtype)))[:, None]
        elif H0_mode == "block_jacobi":
            Minv = diag_u
            def H0_matvec(r):
                u = params
                m = u / jnp.maximum(jnp.linalg.norm(u, axis=1, keepdims=True), jnp.asarray(1e-12, u.dtype))
                r_tan = r - jnp.sum(r * m, axis=1, keepdims=True) * m
                return jnp.einsum("nij,nj->ni", Minv, r_tan)
        else:
            raise ValueError(f"Unknown H0_mode={H0_mode!r}.")

        # Two-loop on -grad
        descent = jtu.tree_map(lambda g: -g, state.grad)
        if solver.history_size:
            start = state.iter_num % solver.history_size
            descent = _inv_hessian_product_H0(descent, state.s_history, state.y_history,
                                              state.rho_history, start, H0_matvec)

        # Ensure descent
        gtd = _tree_vdot(state.grad, descent)
        descent = jax.lax.cond(gtd < 0, lambda _: descent,
                               lambda _: jtu.tree_map(lambda g: -g, state.grad),
                               operand=None)

        # IMPORTANT: recompute g·d for the final direction
        gtd = _tree_vdot(state.grad, descent)  # phi'(0)

        def _increase_mode_init():
            is_first = (state.iter_num == 0)
            base = jax.lax.cond(
                is_first,
                lambda _: jnp.asarray(ls_init_stepsize, dtype=params.dtype),
                lambda _: state.stepsize,
                operand=None)
            return jax.lax.cond(
                is_first,
                lambda _: jnp.minimum(base, solver.max_stepsize),
                lambda _: jnp.minimum(base * jnp.asarray(ls_increase_factor, base.dtype),
                                      solver.max_stepsize),
                operand=None)

        # -----------------------
        # Select initial stepsize
        # -----------------------
        if ls_init_mode == "value" and (ls_init_stepsize is not None):
            init_stepsize = jnp.asarray(ls_init_stepsize, dtype=params.dtype)
        
        elif ls_init_mode == "current":
            is_first = (state.iter_num == 0)
            init_stepsize = jax.lax.cond(
                is_first,
                lambda _: jnp.asarray(ls_init_stepsize, dtype=params.dtype),
                lambda _: state.stepsize,
                operand=None)            
        
        elif ls_init_mode == "alpha0":
            # Quadratic initializer:
            #   alpha0 = min(1, 1.01 * 2*(f_k - f_{k-1}) / (g_k^T p_k))
            # Use "increase" fallback on first iter or invalid ratio.
            base = _increase_mode_init()                 # fallback
            num = 2.0 * (state.value - f_prev)           # <= 0 when f_k <= f_{k-1}
            den = gtd                                    # <= 0 for descent
            # Avoid division by zero; pick a negative placeholder to keep raw invalid if den == 0
            raw = num / jnp.where(den != 0.0, den, jnp.asarray(-1.0, den.dtype))
            # raw is valid iff finite and > 0
            valid = jnp.isfinite(raw) & (raw > 0.0)
            # Nocedal–Wright tweak: min(1, 1.01*raw), then clamp to [1e-16, max_stepsize]
            cand = jnp.minimum(jnp.asarray(1.0, raw.dtype), 1.01 * raw)
            cand = jnp.clip(cand,
                            jnp.asarray(1e-16, cand.dtype),
                            jnp.asarray(solver.max_stepsize, cand.dtype))
            init_stepsize = jax.lax.select(valid, cand, base)

        elif ls_init_mode == "increase":
            init_stepsize = _increase_mode_init()
        
        else:
            # Safe default
            init_stepsize = jnp.asarray(solver.max_stepsize, dtype=params.dtype)

        # Line search

        # --- Build tangent direction for the curvilinear path ---
        d_tan = _apply_precond_tangent(params, descent, precond_mode="none", diag_u=None)

        # --- Curvilinear Armijo backtracking ---
        '''
        (new_stepsize, new_params, new_value, new_grad,
         aux_flat_new, parts_new, ls_iter_num, ls_failed, fe_incr) = _curvilinear_backtracking_armijo(
            fun_value_and_grad,
            params, state.value, state.grad, state.aux,
            d_tan, init_stepsize,
            c1=ls_c1, decrease_factor=ls_decrease, maxiter=60
        )
        '''

        eta1, eta2 = ls_c1, 1.0-ls_c2
        (new_stepsize, new_params, new_value, new_grad,
         aux_flat_new, parts_new, ls_iter_num, ls_failed, fe_incr) = _curvilinear_modified_armijo(
            fun_value_and_grad,
            params, state.value, state.grad, state.aux,
            d_tan, init_stepsize,
            eta1=eta1, eta2=eta2, C=ls_increase, shrink=ls_decrease,
            max_enlarge=20, max_reduce=60, tau_max=ls_max_stepsize
        )

        new_aux = (aux_flat_new, parts_new)

        # Standard L-BFGS curvature pair (no Powell damping; Wolfe LS enforces s^T y > 0)
        s = jtu.tree_map(lambda a, b: a - b, new_params, params)        # s_k = x_{k+1} - x_k
        y = jtu.tree_map(lambda a, b: a - b, new_grad, state.grad)      # y_k = g_{k+1} - g_k

        # Curvature scalar
        sTy = _tree_vdot(s, y)

        # Accept the pair only if it has sufficient positive curvature (Wolfe guarantees this in theory).
        # The tiny threshold avoids numerical issues if s^T y is extremely small.
        eps_curv = jnp.asarray(1e-30, dtype=sTy.dtype)
        write_mask = sTy > eps_curv

        # Standard L-BFGS rho = 1 / (s^T y); set to 0 when we skip writing (keeps state consistent)
        rho = jnp.where(write_mask, 1.0 / sTy, 0.0)

        hist = solver.history_size  # Python int, static for JIT
        s_hist = state.s_history
        y_hist = state.y_history
        rho_hist = state.rho_history

        if hist:
            start = state.iter_num % hist

            # Previous slot values
            s_prev_slot = jtu.tree_map(lambda H: H[start], s_hist)
            y_prev_slot = jtu.tree_map(lambda H: H[start], y_hist)
            rho_prev = rho_hist[start]

            # Conditionally write the new curvature pair
            s_to_write = jtu.tree_map(lambda newv, prev: jax.lax.select(write_mask, newv, prev), s, s_prev_slot)
            y_to_write = jtu.tree_map(lambda newv, prev: jax.lax.select(write_mask, newv, prev), y, y_prev_slot)
            rho_to_write = jax.lax.select(write_mask, rho, rho_prev)

            s_hist = jtu.tree_map(lambda H, v: H.at[start].set(v), s_hist, s_to_write)
            y_hist = jtu.tree_map(lambda H, v: H.at[start].set(v), y_hist, y_to_write)
            rho_hist = rho_hist.at[start].set(rho_to_write)

        gamma_new = jax.lax.select(write_mask, _compute_gamma_scalar(s_hist, y_hist, start), state.gamma)

        if debug:
            jax.debug.print(
                "{:04d} f {:.6e} it {:02d} α0 {:.3e} α {:.3e} {} g·d {:+.3e} γ: {:.3e}",
                state.iter_num, new_value, ls_iter_num, init_stepsize, new_stepsize,
                ls_failed, gtd, gamma_new
            )

        error = jnp.linalg.norm(new_grad, ord=jnp.inf)
        new_state = type(state)(
            iter_num=state.iter_num + 1,
            value=new_value,
            grad=new_grad,
            stepsize=jnp.asarray(new_stepsize, dtype=state.rho_history.dtype),
            error=jnp.asarray(jnp.linalg.norm(new_grad, ord=jnp.inf), dtype=state.rho_history.dtype),
            s_history=s_hist, y_history=y_hist, rho_history=rho_hist,
            gamma=gamma_new,
            aux=new_aux,
            failed_linesearch=ls_failed,
            num_fun_eval=state.num_fun_eval + fe_incr,
            num_grad_eval=state.num_grad_eval + fe_incr,         # value+grad computed together
            num_linesearch_iter=state.num_linesearch_iter + ls_iter_num,
        )
        return new_params, new_state, params, state.value

    params_star, state_star, _, _ = jax.lax.while_loop(cond_fun, body_fun, carry0)
    metrics = jnp.asarray([state_star.iter_num, state_star.num_fun_eval], dtype=jnp.int32)
    return params_star, state_star, metrics

# =============================================================================
# Front-end
# =============================================================================
def minimize_energy_lbfgs(
    initial_u_raw: jnp.ndarray,
    initial_A: Optional[jnp.ndarray],
    A_t, P_t, R_t, Dinv_t, L_c,
    conn: jnp.ndarray, grad_phi: jnp.ndarray,
    volume: jnp.ndarray, mat_id: jnp.ndarray,
    Ms_lookup: jnp.ndarray,
    A_lookup_exchange: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    k_easy_e: jnp.ndarray,
    H_ext: jnp.ndarray,
    E_ref: jnp.ndarray,
    *,
    # Core A/U-solver tuning:
    tol: float = 1e-3, maxiter: int = 500,
    nu_pre: int = 2, nu_post: int = 2, omega: float = 0.7,
    coarse_iters: int = 8, coarse_omega: float = 0.7,
    # Mapping:
    eps_norm: float = 1e-12,
    # Gauge (A only):
    gauge: float = 0.0,
    # Magnetostatics formulation:
    ms_mode: str = "A",
    # Outer LBFGS:
    history_size: int = 10, outer_max_iter: int = 200, grad_tol: float = 1e-3,
    debug_lbfgs: bool = False,
    # Two-loop H0 (no CG):
    H0_mode: str = "gamma",
    # LS knobs:
    ls_init_mode: str = "increase",
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    ls_increase_factor: float = 1.5,
    ls_kind: str = "default",
    ls_c1: float = 1e-4,
    ls_c2: float = 0.9,
    ls_decrease: float = 0.5,
    ls_increase: float = 2.0, # C in modified Armijo
    diag_u=None,  # optional precomputed diag or Minv
):
    N = initial_u_raw.shape[0]
    if ms_mode == "A":
        A0_flat = (jnp.zeros((3 * N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))
    else:
        A0_flat = (jnp.zeros((N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))

    fun_value_and_grad = make_fun_for_jaxopt(
        A_t=A_t, P_t=P_t, R_t=R_t, Dinv_t=Dinv_t, L_c=L_c,
        conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
        Ms_lookup=Ms_lookup, A_lookup_exchange=A_lookup_exchange,
        K1_lookup=K1_lookup, k_easy_e=k_easy_e,
        H_ext=H_ext, E_ref=E_ref,
        tol=tol, maxiter=maxiter, nu_pre=nu_pre, nu_post=nu_post, omega=omega,
        coarse_iters=coarse_iters, coarse_omega=coarse_omega,
        gauge=gauge,
        ms_mode=ms_mode,
        eps_norm=eps_norm,
    )

    u_star, state, metrics = run_jaxopt_lbfgs_twoloop(
        initial_u_raw, A0_flat,
        fun_value_and_grad=fun_value_and_grad,
        history_size=history_size, max_iter=outer_max_iter, grad_tol=grad_tol,
        H0_mode=H0_mode,
        ls_init_mode=ls_init_mode, ls_init_stepsize=ls_init_stepsize,
        ls_max_stepsize=ls_max_stepsize, ls_increase_factor=ls_increase_factor, 
        debug=debug_lbfgs,
        ls_kind=ls_kind,
        ls_c1=ls_c1, ls_c2=ls_c2, ls_decrease=ls_decrease, ls_increase=ls_increase,
        diag_u=diag_u
    )

    E_norm = state.value
    aux_star = state.aux  # (3N,) for A, (N,) for U
    
    aux_flat, parts_norm = aux_star
    if ms_mode == "A":
        A_or_U = aux_flat.reshape((-1, 3))
    else:
        A_or_U = aux_flat.reshape((-1,))
    return E_norm, u_star, A_or_U, parts_norm, state.stepsize, metrics



minimize_energy_lbfgs = jax.jit(
    minimize_energy_lbfgs,
    static_argnames=(
        "tol","maxiter","nu_pre","nu_post","omega","coarse_iters","coarse_omega",
        "eps_norm",
        "history_size","outer_max_iter","grad_tol",
        "debug_lbfgs",
        "H0_mode",
        "ls_init_mode",      # mode -> static
        "ls_max_stepsize",   # constructor knob -> static
        "ls_kind",           # constructor knob -> static
        "ls_c1",             # constructor knob if BacktrackingLineSearch is used -> static
        "ls_c2",             # constructor knob -> static
        "ls_decrease",       # constructor knob -> static
        "ms_mode",           # mode -> static
        # NOTE: ls_init_stepsize intentionally NOT static (dynamic per field)
        # ls_increase_factor can be left dynamic as long as you don't put it in a constructor
    ),
)

# =============================================================================
# Barzilai–Borwein (spectral gradient) driver with Armijo line-search
# =============================================================================

def run_bb_gradient(
    init_params,
    init_aux_flat,
    *,
    fun_value_and_grad,
    max_iter: int,
    grad_tol: float,
    bb_variant: str = "alt",          # "alt", "bb1", "bb2"
    stepsize_init: float = 1.0,
    stepsize_max: float = 1.0,
    stepsize_min: float = 1e-16,
    ls_c1: float = 1e-4,
    ls_decrease: float = 0.5,
    debug: bool = False,
    init_ls_iters: int = 2,
):
    """
    Curvilinear Barzilai–Borwein (curveBB) with warm-up:
      - First 'init_ls_iters' steps: curvilinear Armijo LS (Algorithm 1).
      - Afterwards: pure BB steps along the curvilinear path (Algorithm 2), NO line search.
    """
    # Initial evaluation (value+grad, plus aux with parts)
    (f0, (aux_flat0, parts0)), g0 = fun_value_and_grad(init_params, init_aux_flat)
    params0 = _normalize_u(init_params)   # store normalized; curvilinear updates preserve unit length
    aux_args0 = (aux_flat0,)

    # Tolerances, counters, bookkeeping
    TauF = jnp.asarray(grad_tol, dtype=params0.dtype)
    TauF_sqrt = jnp.sqrt(TauF)
    TauF_cuberoot = jnp.power(TauF, 1.0 / 3.0)
    epsM = jnp.asarray(jnp.finfo(params0.dtype).eps, dtype=params0.dtype)

    x_prev = params0
    g_prev = g0
    f_prev = f0
    alpha_prev = jnp.asarray(stepsize_init, dtype=params0.dtype)
    it0 = jnp.asarray(0, jnp.int32)
    fe_total0 = jnp.asarray(1, jnp.int32)           # we already did one evaluation
    warm_left0 = jnp.asarray(init_ls_iters, jnp.int32)

    # Carry: (params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total, warm_left)
    carry0 = (params0, f0, g0, aux_args0, x_prev, g_prev, f_prev, alpha_prev, it0, parts0, fe_total0, warm_left0)

    def cond_fun(carry):
        params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total, warm_left = carry
        x_norm = jnp.linalg.norm(params, ord=jnp.inf)
        g_norm = jnp.linalg.norm(g_k, ord=jnp.inf)
        step_norm = jnp.linalg.norm(params - x_prev, ord=jnp.inf)

        eta_hat = TauF * (1.0 + jnp.abs(f_k))
        eps_A = epsM * (1.0 + jnp.abs(f_k))

        fun_ok  = jax.lax.cond(it > 0, lambda _: jnp.abs(f_prev - f_k) <= eta_hat, lambda _: False, operand=None)
        step_ok = jax.lax.cond(it > 0, lambda _: step_norm <= (TauF_sqrt * (1.0 + x_norm)), lambda _: False, operand=None)
        grad_ok = g_norm <= (TauF_cuberoot * (1.0 + jnp.abs(f_k)))
        grad_abs_ok = g_norm < eps_A

        success = (fun_ok & step_ok & grad_ok) | grad_abs_ok
        return jnp.logical_and(~success, it < max_iter)

    def body_fun(carry):
        params, f_k, g_k, aux_args, x_prev, g_prev, f_prev, alpha_prev, it, parts_last, fe_total, warm_left = carry

        # Tangent descent direction
        Pg = _apply_precond_tangent(params, g_k, precond_mode='none', diag_u=None)
        d_tan = jtu.tree_map(lambda x: -x, Pg)

        # --- BB stepsize (only used when warm_left == 0) ---
        s = jtu.tree_map(lambda a, b: a - b, params, x_prev)
        y = jtu.tree_map(lambda a, b: a - b, g_k, g_prev)
        sty = _tree_vdot(s, y)
        sts = _tree_vdot(s, s)
        yty = _tree_vdot(y, y)

        alpha_bb1 = sts / jnp.maximum(sty, jnp.asarray(1e-30, dtype=sts.dtype))
        alpha_bb2 = sty / jnp.maximum(yty, jnp.asarray(1e-30, dtype=yty.dtype))

        use_bb1 = (bb_variant == "bb1") | ((bb_variant == "alt") & ((it % 2) == 0))
        alpha_raw = jnp.where(use_bb1, alpha_bb1, alpha_bb2)

        # Clamp and guard (positive, finite); fall back to previous on invalid
        def _is_valid(a): return jnp.isfinite(a) & (a > 0.0)
        alpha_guess = jax.lax.cond(_is_valid(alpha_raw), lambda _: alpha_raw, lambda _: alpha_prev, operand=None)
        alpha_guess = jnp.clip(alpha_guess,
                               jnp.asarray(stepsize_min, alpha_guess.dtype),
                               jnp.asarray(stepsize_max, alpha_guess.dtype))

        # --- Two modes: warm-up LS (Algorithm 1) vs BB step without LS (Algorithm 2) ---
        def _warmup_ls(_):

            return _curvilinear_backtracking_armijo(
                fun_value_and_grad,
                params, f_k, g_k, aux_args[0],
                d_tan,
                tau_init=jnp.asarray(stepsize_init, params.dtype),
                c1=ls_c1,
                decrease_factor=ls_decrease,
                maxiter=60
            )

            '''
            return _curvilinear_modified_armijo(
                fun_value_and_grad,
                params, f_k, g_k, aux_args[0],
                d_tan,
                tau_init=jnp.asarray(stepsize_init, params.dtype),
                eta1=0.3, eta2=0.3, C=ls_increase_factor, shrink=ls_decrease,
                max_enlarge=20, max_reduce=60, tau_max=stepsize_max,
            )
            '''

        def _bb_no_ls(_):
            # Take a single curvilinear step with BB stepsize (no LS)
            trial_params = _curvilinear_step(params, d_tan, alpha_guess)
            (trial_value, (aux_flat_new, parts_new)), trial_grad = fun_value_and_grad(trial_params, aux_args[0])
            fe_incr = jnp.asarray(1, jnp.int32)
            return (alpha_guess, trial_params, trial_value, trial_grad,
                    aux_flat_new, parts_new,
                    jnp.asarray(0, jnp.int32), jnp.asarray(False, jnp.bool_), fe_incr)

        (new_stepsize, new_params, new_value, new_grad,
         aux_flat_new, parts_new, ls_iter_num, ls_failed, fe_incr) = jax.lax.cond(
            warm_left > 0, _warmup_ls, _bb_no_ls, operand=None
        )
        aux_args_next = (aux_flat_new,)

        # Optional debug
        if debug:
            gtd = _tree_vdot(g_k, d_tan)
            mode_flag = jnp.where(warm_left > 0, jnp.asarray(1, jnp.int32), jnp.asarray(0, jnp.int32))
            jax.debug.print(
                " [BB mode={}] it {:02d} f {:.6e} tau0 {:.3e} tau {:.3e} ls_it {:02d} failed:{} g·d {:+.3e}",
                mode_flag, it, new_value, alpha_prev, new_stepsize, ls_iter_num, ls_failed, gtd
            )

        fe_total_next = fe_total + fe_incr
        # Decrement warm-up counter only if used warm-up this iteration
        warm_next = jax.lax.select(warm_left > 0, jnp.maximum(warm_left - 1, 0), warm_left)

        # Advance iteration
        return (new_params,          # params
                new_value,           # f_k
                new_grad,            # g_k
                aux_args_next,       # aux_args
                params,              # x_prev
                g_k,                 # g_prev
                f_k,                 # f_prev
                new_stepsize,        # alpha_prev
                it + 1,              # it
                parts_new,           # parts_last
                fe_total_next,       # fe_total
                warm_next)           # warm_left

    params_star, f_star, g_star, aux_args_star, _, _, _, step_size, it_star, parts_last, fe_total_star, _ = \
        jax.lax.while_loop(cond_fun, body_fun, carry0)

    # Compose aux to match the LBFGS/BB front-end return
    aux_flat_star = aux_args_star[0]
    aux_star = (aux_flat_star, parts_last)

    class _State:
        pass
    st = _State()
    st.value = f_star
    st.grad = g_star
    st.aux = aux_star
    st.iter_num = 0
    st.last_stepsize = step_size

    metrics = jnp.asarray([it_star, fe_total_star], dtype=jnp.int32)
    return params_star, st, metrics

# =============================================================================
# Front-end wrapper: same signature & return shape as minimize_energy_lbfgs
# =============================================================================
def minimize_energy_bb(
    initial_u_raw: jnp.ndarray,
    initial_A: Optional[jnp.ndarray],
    A_t, P_t, R_t, Dinv_t, L_c,
    conn: jnp.ndarray, grad_phi: jnp.ndarray,
    volume: jnp.ndarray, mat_id: jnp.ndarray,
    Ms_lookup: jnp.ndarray,
    A_lookup_exchange: jnp.ndarray,
    K1_lookup: jnp.ndarray,
    k_easy_e: jnp.ndarray,
    H_ext: jnp.ndarray,
    E_ref: jnp.ndarray,
    *,
    # Core A/U-solver tuning:
    tol: float = 1e-3, maxiter: int = 500,
    nu_pre: int = 2, nu_post: int = 2, omega: float = 0.7,
    coarse_iters: int = 8, coarse_omega: float = 0.7,
    # Mapping:
    eps_norm: float = 1e-12,
    # Gauge (A only):
    gauge: float = 0.0,
    # Magnetostatics formulation:
    ms_mode: str = "A",
    # Outer BB:
    outer_max_iter: int = 200, grad_tol: float = 1e-3,
    bb_variant: str = "alt",             # "alt", "bb1", "bb2"
    debug_bb: bool = False,
    # Line-search knobs:
    ls_init_stepsize: float = 1.0,
    ls_max_stepsize: float = 1.0,
    ls_decrease: float = 0.5,
    ls_c1: float = 1.0e-4,
    bb_init_steps=2,
):
    N = initial_u_raw.shape[0]
    if ms_mode == "A":
        A0_flat = (jnp.zeros((3 * N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))
    else:
        A0_flat = (jnp.zeros((N,), dtype=jnp.float64)
                   if initial_A is None else initial_A.reshape((-1,)))

    fun_value_and_grad = make_fun_for_jaxopt(
        A_t=A_t, P_t=P_t, R_t=R_t, Dinv_t=Dinv_t, L_c=L_c,
        conn=conn, grad_phi=grad_phi, volume=volume, mat_id=mat_id,
        Ms_lookup=Ms_lookup, A_lookup_exchange=A_lookup_exchange,
        K1_lookup=K1_lookup, k_easy_e=k_easy_e,
        H_ext=H_ext, E_ref=E_ref,
        tol=tol, maxiter=maxiter, nu_pre=nu_pre, nu_post=nu_post, omega=omega,
        coarse_iters=coarse_iters, coarse_omega=coarse_omega,
        gauge=gauge,
        ms_mode=ms_mode,
        eps_norm=eps_norm,
    )

    u_star, state, metrics = run_bb_gradient(
        initial_u_raw, A0_flat,
        fun_value_and_grad=fun_value_and_grad,
        max_iter=outer_max_iter, grad_tol=grad_tol,
        bb_variant=bb_variant,
        stepsize_init=ls_init_stepsize,
        stepsize_max=ls_max_stepsize,
        stepsize_min=1e-16,
        ls_c1=ls_c1,
        ls_decrease=ls_decrease,
        debug=debug_bb,
        init_ls_iters=bb_init_steps,
    )

    E_norm = state.value
    aux_flat, parts_norm = state.aux  # <- already a (aux_flat, parts_norm) tuple now
    if ms_mode == "A":
        A_or_U = aux_flat.reshape((-1, 3))
    else:
        A_or_U = aux_flat.reshape((-1,))
    return E_norm, u_star, A_or_U, parts_norm, state.last_stepsize, metrics


minimize_energy_bb = jax.jit(
    minimize_energy_bb,
    static_argnames=(
        "tol", "maxiter", "nu_pre", "nu_post", "omega", "coarse_iters", "coarse_omega",
        "eps_norm",
        "outer_max_iter", "grad_tol",
        "debug_bb",       # you can keep this static; toggling it would retrace
        "bb_variant",     # control-flow/mode -> static
        "ms_mode",        # control-flow/mode -> static
        # NOTE: ls_init_stepsize is intentionally dynamic
        # NOTE: ls_max_stepsize can be dynamic; it is used only for clamping
        # NOTE: ls_print can be left out; see note below
    ),
)

# =============================================================================
# Helper: uniform u_raw init (from InitialState mx,my,mz)
# =============================================================================
def _normalize_initial_xyz(mx: float, my: float, mz: float, eps: float = 1e-30) -> jnp.ndarray:
    v = jnp.asarray([mx, my, mz], dtype=jnp.float64)
    n = jnp.linalg.norm(v)
    def _unit(_): return v / jnp.maximum(n, jnp.asarray(eps, v.dtype))
    def _fallback(_): return jnp.array([0.0, 0.0, 1.0], dtype=jnp.float64)
    return jax.lax.cond(n > eps, _unit, _fallback, operand=None)


def make_uniform_u_raw(N: int, init: Any, scale: float = 1.0) -> jnp.ndarray:
    mx = jnp.asarray(init.mx, dtype=jnp.float64)
    my = jnp.asarray(init.my, dtype=jnp.float64)
    mz = jnp.asarray(init.mz, dtype=jnp.float64)
    v_hat = _normalize_initial_xyz(mx, my, mz)
    return jnp.tile((scale * v_hat)[None, :], (N, 1))
