#!/usr/bin/env python3
"""
Magnetostatics core (cleaned):
- Preprocessing helpers used by loop.py
- AMG hierarchy builder used by optimize.py / loop.py
- JAX CG core (jitted) for solving the A-field used by both
"""
from __future__ import annotations
from jax import config
config.update("jax_enable_x64", True)
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import configparser
from dataclasses import dataclass
from jax.experimental.sparse import BCOO
import jax.scipy.sparse.linalg as jsp_linalg
from jax.scipy.linalg import solve_triangular
from add_shell import run_add_shell_pipeline
from geom import precompute_geometry_from_knt_ijk, TetGeom
from energies import MU0, brown_energy_and_grad_from_m, brown_energy_and_grad_from_scalar_potential 
# (Removed global GAUGE)

# -------------------- AMG hierarchy cache & types --------------------
_MG_CACHE: Dict[str, Dict[str, Any]] = {}

@dataclass(frozen=True)
class JaxAmgHierarchy:
    # Tuples for static shapes in JIT (fine -> coarse)
    A: tuple      # tuple[BCOO], length L
    P: tuple      # tuple[BCOO], length L-1
    R: tuple      # tuple[BCOO], length L-1
    Dinv: tuple   # tuple[jnp.ndarray], length L (1/diag(A_l))


def _mesh_cache_key(geom: TetGeom, gauge: float, amg: str) -> str:
    conn_np = np.asarray(geom.conn, dtype=np.int32)
    h = hashlib.sha1(conn_np.tobytes()).hexdigest()
    gkey = f"{gauge:.12e}"
    return f"amg={amg}\ngauge={gkey}\nE={conn_np.shape[0]}\nn={conn_np.shape[1]}\n{h}"


# Assemble scalar Laplacian (CPU/SciPy) used to build the JAX hierarchy
from amg_pcg import _assemble_scalar_laplacian as assemble_scalar_laplacian  # used here

def _build_jax_amg_hierarchy_from_geom(geom: TetGeom, amg: str = "sa", gauge: float = 0.0) -> JaxAmgHierarchy:
    """
    Build a JAX-side AMG hierarchy from geometry.
    Scales by 1/mu0 and shifts by 'gauge' on the finest level.
    """
    # Assemble scalar Laplacian (CPU, SciPy)
    conn_np = np.asarray(geom.conn, dtype=np.int32)
    grad_phi = np.asarray(geom.grad_phi, dtype=np.float64)
    volume = np.asarray(geom.volume, dtype=np.float64)
    N = int(np.max(conn_np) + 1)
    L_csr = assemble_scalar_laplacian(N, conn_np, grad_phi, volume)

    # Physics scaling
    L_csr = (1.0 / MU0) * L_csr
    # Finest-level shift: gauge * I
    if gauge and float(gauge) != 0.0:
        import scipy.sparse as sp
        L_csr = L_csr + gauge * sp.eye(N, dtype=np.float64, format="csr")

    # Build PyAMG hierarchy (CPU) -> convert each level to JAX BCOO + Dinv
    try:
        import pyamg
    except Exception as exc:
        raise RuntimeError("pyamg is required for building AMG hierarchy (pip install pyamg)") from exc

    if amg == "sa":
        ml = pyamg.smoothed_aggregation_solver(L_csr, symmetry='symmetric')
    elif amg == "rs":
        ml = pyamg.ruge_stuben_solver(L_csr, symmetry='symmetric')
    else:
        raise ValueError("amg must be 'sa' or 'rs'")

    A_list, P_list, R_list, Dinv_list = [], [], [], []
    for lvl in ml.levels:
        A_csr = lvl.A.tocsr()
        # Dinv = 1 / diag(A) with small ridge for safety
        Dinv_list.append(jax.device_put(1.0 / (A_csr.diagonal().astype(np.float64) + 1e-30)))
        A_list.append(BCOO.from_scipy_sparse(A_csr))
    for i in range(len(ml.levels) - 1):
        P_csr = ml.levels[i].P.tocsr()
        R_csr = P_csr.T.tocsr()
        P_list.append(BCOO.from_scipy_sparse(P_csr))
        R_list.append(BCOO.from_scipy_sparse(R_csr))

    return JaxAmgHierarchy(
        A=tuple(A_list),
        P=tuple(P_list),
        R=tuple(R_list),
        Dinv=tuple(Dinv_list),
    )


def get_or_build_jax_amg_hierarchy(geom: TetGeom, amg: str = "sa", gauge: float = 0.0) -> JaxAmgHierarchy:
    key = _mesh_cache_key(geom, gauge, amg)
    entry = _MG_CACHE.get(key, {})
    if "hier" in entry:
        return entry["hier"]
    hier = _build_jax_amg_hierarchy_from_geom(geom, amg=amg, gauge=gauge)
    entry["hier"] = hier
    _MG_CACHE[key] = entry
    return hier

def _mesh_cache_key_scalar(geom: TetGeom, amg: str) -> str:
    """Cache key for scalar AMG (built on μ0*K)."""
    conn_np = np.asarray(geom.conn, dtype=np.int32)
    h = hashlib.sha1(conn_np.tobytes()).hexdigest()
    return f"kind=scalar\namg={amg}\nmu0_scaled=1\nE={conn_np.shape[0]}\nn={conn_np.shape[1]}\n{h}"


def _build_jax_amg_hierarchy_for_scalar(
    geom: TetGeom,
    amg: str = "sa",
) -> JaxAmgHierarchy:
    """
    Build a JAX-side AMG hierarchy for the scalar operator A = μ0*K.
    - K is the standard FEM scalar Laplacian (Eq. 43).
    - No gauge shift is added for the scalar potential.
    - Scaling by μ0 is baked into the hierarchy so the V-cycle approximates (μ0*K)^{-1}.
    """
    # Assemble scalar Laplacian K (CPU, SciPy)
    conn_np = np.asarray(geom.conn, dtype=np.int32)
    grad_phi = np.asarray(geom.grad_phi, dtype=np.float64)
    volume = np.asarray(geom.volume, dtype=np.float64)
    N = int(np.max(conn_np) + 1)

    # Uses the same assembler as the A-field block (already imported at module scope)
    L_csr = assemble_scalar_laplacian(N, conn_np, grad_phi, volume)  # K

    # Scale to μ0*K for the scalar CG operator
    L_csr = float(MU0) * L_csr

    # Build PyAMG hierarchy (CPU) → convert to JAX BCOO + Dinv
    try:
        import pyamg
    except Exception as exc:
        raise RuntimeError("pyamg is required for building AMG hierarchy (pip install pyamg)") from exc

    if amg == "sa":
        ml = pyamg.smoothed_aggregation_solver(L_csr, symmetry='symmetric')
    elif amg == "rs":
        ml = pyamg.ruge_stuben_solver(L_csr, symmetry='symmetric')
    else:
        raise ValueError("amg must be 'sa' or 'rs'")

    A_list, P_list, R_list, Dinv_list = [], [], [], []
    for lvl in ml.levels:
        A_csr = lvl.A.tocsr()
        Dinv_list.append(jax.device_put(1.0 / (A_csr.diagonal().astype(np.float64) + 1e-30)))
        A_list.append(BCOO.from_scipy_sparse(A_csr))

    for i in range(len(ml.levels) - 1):
        P_csr = ml.levels[i].P.tocsr()
        R_csr = P_csr.T.tocsr()
        P_list.append(BCOO.from_scipy_sparse(P_csr))
        R_list.append(BCOO.from_scipy_sparse(R_csr))

    return JaxAmgHierarchy(
        A=tuple(A_list),
        P=tuple(P_list),
        R=tuple(R_list),
        Dinv=tuple(Dinv_list),
    )


def get_or_build_jax_amg_hierarchy_for_scalar(
    geom: TetGeom,
    amg: str = "sa",
) -> JaxAmgHierarchy:
    """
    Cached builder for the scalar AMG hierarchy (on μ0*K).
    """
    key = _mesh_cache_key_scalar(geom, amg)
    entry = _MG_CACHE.get(key, {})
    if "hier" in entry:
        return entry["hier"]
    hier = _build_jax_amg_hierarchy_for_scalar(geom, amg=amg)
    entry["hier"] = hier
    _MG_CACHE[key] = entry
    return hier

def energy_and_grad_U(
    m_nodes: jnp.ndarray,
    u_nodes: jnp.ndarray,
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Returns (S, grad_wrt_m, grad_wrt_u) for Brown's scalar-potential functional (Eq. 38).
    Thin wrapper around energies.brown_energy_and_grad_from_scalar_potential(...).
    """
    return brown_energy_and_grad_from_scalar_potential(
        m_nodes=m_nodes,
        u_nodes=u_nodes,
        geom=geom,
        Ms_lookup=Ms_lookup,
    )

# -------------------- Energy & gradient adapter (Brown functional) --------------------
def energy_and_grad_A(m_nodes: jnp.ndarray,
                      A_nodes: jnp.ndarray,
                      geom: TetGeom,
                      Ms_lookup: jnp.ndarray) -> Tuple[jnp.ndarray, Any, jnp.ndarray]:
    """
    Returns (S, aux, grad_wrt_A) using the Brown functional.
    Thin wrapper around energies.brown_energy_and_grad_from_m(...)
    """
    return brown_energy_and_grad_from_m(
        m_nodes=m_nodes,
        A_nodes=A_nodes,
        geom=geom,
        Ms_lookup=Ms_lookup
    )


fl = lambda x: x.reshape(-1)
un = lambda y: y.reshape((-1, 3))


# -------------------- JAX CG + Per-component V-cycle preconditioner --------------------
def _solve_A_jax_cg_compMG_core(
    A_t, P_t, R_t, Dinv_t, L_c,  # tuples (hier parts) + coarse Cholesky factor
    x0_flat: jnp.ndarray,        # initial guess (flattened, shape 3N,)
    m_nodes: jnp.ndarray,
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
    tol: float,
    maxiter: int,
    *,
    gauge: float = 0.0,  # <--- NEW: gauge passed in explicitly (no global)
    nu_pre: int = 2,
    nu_post: int = 2,
    omega: float = 0.7,
    coarse_iters: int = 8,
    coarse_omega: float = 0.7,
):
    """
    JAX core for PCG with a per-component V-cycle M built from tuple components.
    Parameters
    ----------
    x0_flat : jnp.ndarray
        Initial guess for the vector potential, flattened shape (3*N,).
    """
    # ---- local helpers (pure JAX) ----
    def _spmv(A, x):
        return A @ x

    def _jacobi_relax(A, Dinv, x, b, iters, omg):
        def body_fun(i, xk):
            r = b - _spmv(A, xk)
            return xk + omg * (Dinv * r)
        return jax.lax.fori_loop(0, iters, body_fun, x)

    def _coarse_solve(L, b):
        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(L.T, y, lower=False)
        return x

    def _vcycle_scalar(A_list, P_list, R_list, Dinv_list, b,
                   nu_pre, nu_post, omg, coarse_iters, coarse_omg):
        """
        One scalar V-cycle returning x ≈ A^{-1} b on the finest level.
    
        Expects:
          - A_list, P_list, R_list, Dinv_list: AMG tuples from finest->coarsest
          - b: RHS on the finest grid
          - nu_pre, nu_post: number of Jacobi smoothing steps
          - omg: Jacobi relaxation parameter
          - coarse_iters, coarse_omg: fallback Jacobi parameters on the coarsest level
    
        Notes:
          - Uses helpers `_spmv`, `_jacobi_relax`, `_coarse_solve` and a
            closure variable `L_c` (optional coarse Cholesky) from the outer scope.
        """
        L = len(A_list)
        xs, bs = [], []
  
        # Finest level
        x = jnp.zeros_like(b)
        b_l = b
    
        # Down-sweep: levels 0 .. L-2
        for l in range(L - 1):
            A_l, Dinv_l = A_list[l], Dinv_list[l]
            # Pre-smoothing
            x = _jacobi_relax(A_l, Dinv_l, x, b_l, nu_pre, omg)
            # Restrict residual
            r = b_l - _spmv(A_l, x)
            xs.append(x)
            bs.append(b_l)
            b_l = R_list[l] @ r
            # Initialize next-level solution with zero
            x = jnp.zeros_like(b_l)
  
        # Coarse solve: exact if L_c provided, otherwise Jacobi fallback
        if L_c is not None:
            x = _coarse_solve(L_c, b_l)
        else:
            A_c, Dinv_c = A_list[-1], Dinv_list[-1]
            x = _jacobi_relax(A_c, Dinv_c, x, b_l, coarse_iters, coarse_omg)

        # Up-sweep: levels L-2 .. 0
        for l in range(L - 2, -1, -1):
            # Prolongate and correct
            x = xs[l] + (P_list[l] @ x)
            A_l, Dinv_l, b_l = A_list[l], Dinv_list[l], bs[l]
            # Post-smoothing
            x = _jacobi_relax(A_l, Dinv_l, x, b_l, nu_post, omg)

        return x

    def _make_per_component_vcycle_M(A_list, P_list, R_list, Dinv_list,
                                     nu_pre, nu_post, omg, coarse_iters, coarse_omg):
        # Returns a cg-compatible callable M(v_flat)
        def M(v_flat: jnp.ndarray) -> jnp.ndarray:
            V = v_flat.reshape((-1, 3))
            x0 = _vcycle_scalar(A_list, P_list, R_list, Dinv_list, V[:, 0],
                                nu_pre, nu_post, omega, coarse_iters, coarse_omg)
            x1 = _vcycle_scalar(A_list, P_list, R_list, Dinv_list, V[:, 1],
                                nu_pre, nu_post, omega, coarse_iters, coarse_omg)
            x2 = _vcycle_scalar(A_list, P_list, R_list, Dinv_list, V[:, 2],
                                nu_pre, nu_post, omega, coarse_iters, coarse_omg)
            return jnp.stack([x0, x1, x2], axis=1).reshape((-1,))
        return M

    # ---- RHS: b = - grad_A S_{A=0} ----
    N = int(m_nodes.shape[0])
    A_zero = jnp.zeros((N, 3), dtype=jnp.float64)
    _, _, g0 = energy_and_grad_A(m_nodes, A_zero, geom, Ms_lookup)
    b_flat = (-g0).reshape(-1)

    def A_apply(vec_flat: jnp.ndarray) -> jnp.ndarray:
        A = un(vec_flat)
        _, _, gA = energy_and_grad_A(m_nodes, A, geom, Ms_lookup)
        # Stabilize with explicit 'gauge' (same value used in AMG build)
        return fl(gA - g0 + gauge * A)

    # Per-component V-cycle preconditioner constructed from tuple components
    M = _make_per_component_vcycle_M(
        A_t, P_t, R_t, Dinv_t,
        nu_pre, nu_post, omega, coarse_iters, coarse_omega
    )

    # Use provided initial guess x0_flat
    x_flat, info = jsp_linalg.cg(
        A_apply, b_flat, x0=x0_flat, tol=tol, atol=0.0, maxiter=maxiter, M=M
    )

    # Diagnostics (device)
    A_sol = un(x_flat)
    S_final, _, _ = energy_and_grad_A(m_nodes, A_sol, geom, Ms_lookup)
    iters = jnp.int32(-1)  # iteration counting can be added if desired
    r_flat = A_apply(x_flat) - b_flat
    rnorm = jnp.linalg.norm(r_flat)
    return A_sol, iters, rnorm, S_final, info


_solve_A_jax_cg_compMG_core_jit = jax.jit(
    _solve_A_jax_cg_compMG_core,
    static_argnames=("maxiter", "nu_pre", "nu_post", "coarse_iters")
)

def _solve_U_jax_cg_compMG_core(
    A_t, P_t, R_t, Dinv_t, L_c,      # tuples (AMG levels) + optional coarse Cholesky factor
    x0: jnp.ndarray,                 # initial guess for u, shape (N,)
    m_nodes: jnp.ndarray,
    geom: TetGeom,
    Ms_lookup: jnp.ndarray,
    tol: float,
    maxiter: int,
    *,
    nu_pre: int = 2,
    nu_post: int = 2,
    omega: float = 0.7,
    coarse_iters: int = 8,
    coarse_omega: float = 0.7,
):
    """
    PCG core for the scalar-potential system with a scalar V-cycle AMG preconditioner.

    Operator (SPD):
        A(u) = -(∂S/∂u(u) - ∂S/∂u(0)) = μ0 * K * u
    Right-hand side (IMPORTANT: positive sign):
        b = +∂S/∂u(0) = μ0 * g

    With this choice, the linear system is: μ0 K u = μ0 g  ->  K u = g,
    which yields H = -∇u with the correct physical sign.
    """
    # ---- local helpers (pure JAX) ----
    def _spmv(A, x):
        return A @ x

    def _jacobi_relax(A, Dinv, x, b, iters, omg):
        def body_fun(i, xk):
            r = b - _spmv(A, xk)
            return xk + omg * (Dinv * r)
        return jax.lax.fori_loop(0, iters, body_fun, x)

    def _coarse_solve(L, b):
        y = solve_triangular(L, b, lower=True)
        x = solve_triangular(L.T, y, lower=False)
        return x
    def _vcycle_scalar(A_list, P_list, R_list, Dinv_list, b,
                       nu_pre, nu_post, omg, coarse_iters, coarse_omg):
        """One scalar V-cycle returning x ≈ A^{-1} b (finest level)."""
        L = len(A_list)
        xs, bs = [], []

        x = jnp.zeros_like(b)
        b_l = b
        # Down-sweep: 0 .. L-2
        for l in range(L - 1):
            A_l, Dinv_l = A_list[l], Dinv_list[l]
            x = _jacobi_relax(A_l, Dinv_l, x, b_l, nu_pre, omg)
            r = b_l - _spmv(A_l, x)
            xs.append(x)
            bs.append(b_l)
            b_l = R_t[l] @ r
            x = jnp.zeros_like(b_l)

        # Coarse solve
        if L_c is not None:
            x = _coarse_solve(L_c, b_l)
        else:
            A_c, Dinv_c = A_list[-1], Dinv_list[-1]
            x = _jacobi_relax(A_c, Dinv_c, x, b_l, coarse_iters, coarse_omg)

        # Up-sweep: L-2 .. 0
        for l in range(L - 2, -1, -1):
            x = xs[l] + (P_t[l] @ x)
            A_l, Dinv_l, b_l = A_list[l], Dinv_list[l], bs[l]
            x = _jacobi_relax(A_l, Dinv_l, x, b_l, nu_post, omg)
        return x

    def _make_scalar_vcycle_M(A_list, P_list, R_list, Dinv_list,
                              nu_pre, nu_post, omg, coarse_iters, coarse_omg):
        # Returns a cg-compatible callable M(v) for scalar vectors (N,)
        def M(v: jnp.ndarray) -> jnp.ndarray:
            return _vcycle_scalar(A_list, P_list, R_list, Dinv_list, v,
                                  nu_pre, nu_post, omg, coarse_iters, coarse_omg)
        return M

    # ---- Build RHS and linear operator from scalar energy functional ----
    N = int(m_nodes.shape[0])
    u_zero = jnp.zeros((N,), dtype=jnp.float64)
    _, _, g0 = energy_and_grad_U(m_nodes, u_zero, geom, Ms_lookup)  # grad_u at u=0
    b = g0

    def A_apply(u_vec: jnp.ndarray) -> jnp.ndarray:
        _, _, g_u = energy_and_grad_U(m_nodes, u_vec, geom, Ms_lookup)
        # A(u) = -(g_u - g0)  == μ0*K*u
        return -(g_u - g0)

    # ---- Preconditioner: AMG on A = μ0*K (no extra scaling needed) ----
    M = _make_scalar_vcycle_M(
        A_t, P_t, R_t, Dinv_t,
        nu_pre, nu_post, omega, coarse_iters, coarse_omega,
    )

    # ---- Run CG ----
    x_sol, info = jsp_linalg.cg(A_apply, b, x0=x0, tol=tol, atol=0.0, maxiter=maxiter, M=M)

    # Diagnostics (device-side)
    S_final, _, _ = energy_and_grad_U(m_nodes, x_sol, geom, Ms_lookup)
    iters = jnp.int32(-1)  # optional
    r = A_apply(x_sol) - b
    rnorm = jnp.linalg.norm(r)
    return x_sol, iters, rnorm, S_final, info


_solve_U_jax_cg_compMG_core_jit = jax.jit(
    _solve_U_jax_cg_compMG_core,
    static_argnames=("maxiter", "nu_pre", "nu_post", "coarse_iters"),
)


# -------------------- Preprocessing helpers (loop.py) --------------------
def ensure_npz_suffix(path: str) -> str:
    p = Path(path)
    return str(p.with_suffix(".npz")) if p.suffix == "" else str(p)


def p2_path_for_mesh(mesh_path: str) -> str:
    return str(Path(mesh_path).with_suffix(".p2"))


def read_mesh_params_from_p2(p2_path: str) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Read [mesh] 'size' (default 1.0e-9 if missing) and optional K/KL from a .p2 file.
    Returns (size, K_opt, KL_opt).
    """
    DEFAULT_SIZE = 1.0e-9
    size: float = DEFAULT_SIZE
    K_opt: Optional[float] = None
    KL_opt: Optional[float] = None

    p = Path(p2_path)
    if not p.exists():
        return float(size), K_opt, KL_opt

    cfg = configparser.ConfigParser()
    try:
        with p.open("r") as f:
            cfg.read_file(f)
        if cfg.has_section("mesh"):
            if cfg.has_option("mesh", "size"):
                try:
                    size = cfg.getfloat("mesh", "size", fallback=DEFAULT_SIZE)
                except Exception:
                    size = DEFAULT_SIZE
            if cfg.has_option("mesh", "K"):
                try:
                    K_opt = cfg.getfloat("mesh", "K")
                except Exception:
                    K_opt = None
            if cfg.has_option("mesh", "KL"):
                try:
                    KL_opt = cfg.getfloat("mesh", "KL")
                except Exception:
                    KL_opt = None
    except Exception:
        size, K_opt, KL_opt = DEFAULT_SIZE, None, None
    return float(size), K_opt, KL_opt


def prepare_shells_and_geom(
    *,
    mesh: str,
    K: Optional[float] = 1.5,
    KL: Optional[float] = 5.0,
    auto_layers: bool = True,
    verbose: bool = False,
) -> Tuple[jnp.ndarray, TetGeom, Dict[str, Any]]:
    """
    Reads the BODY mesh, reads `.p2` (size and optional K/KL), generates air shells
    in memory, and precomputes geometry. Returns (knt, geom[TetGeom], meta).

    New behavior:
    - Compute volume_scalefactor = 1 / (volume of mesh WITHOUT the air shell).
    - Multiply geom.volume by this factor.
    - Store the factor in geom.volume_scalefactor.
    """
    mesh = ensure_npz_suffix(mesh)

    # size (and optional K/KL) from .p2
    p2_path = p2_path_for_mesh(mesh)
    size, K_p2, KL_p2 = read_mesh_params_from_p2(p2_path)
    if K is None and K_p2 is not None:
        K = K_p2
    if KL is None and KL_p2 is not None:
        KL = KL_p2
    if not (K > 1.0 and KL > 1.0):
        raise ValueError("K and KL must be > 1.0")

    # Generate shells in-memory (no files written)
    knt_all, ijk_all = run_add_shell_pipeline(
        in_npz=mesh, K=K, KL=KL, auto_layers=auto_layers, verbose=verbose
    )

    # Precompute geometry from in-memory arrays
    knt, geom_raw = precompute_geometry_from_knt_ijk(
        knt_all, ijk_all, detect_one_based=None, fix_orientation=True
    )

    # Compute volume of the mesh WITHOUT air shell
    # Convention: air is the largest material id (G)
    mat_id_raw = jnp.asarray(geom_raw.mat_id, dtype=jnp.int32)
    air_id = jnp.max(mat_id_raw)
    vols_raw = jnp.asarray(geom_raw.volume, dtype=jnp.float64)
    body_vol_sum = jnp.sum(jnp.where(mat_id_raw != air_id, vols_raw, 0.0))

    # volume_scalefactor = 1 / body_vol_sum (safe guard against zero)
    volume_scalefactor = jnp.where(body_vol_sum > 0.0,
                                   1.0 / body_vol_sum,
                                   jnp.asarray(1.0, dtype=jnp.float64))

    # Canonical TetGeom with stable dtypes (good JAX treedef)
    geom = TetGeom(
        conn=jnp.asarray(geom_raw.conn, dtype=jnp.int32),
        grad_phi=jnp.asarray(geom_raw.grad_phi, dtype=jnp.float64),
        volume=jnp.asarray(geom_raw.volume, dtype=jnp.float64) * volume_scalefactor,
        mat_id=jnp.asarray(geom_raw.mat_id, dtype=jnp.int32),
        volume_scalefactor=jnp.asarray(volume_scalefactor, dtype=jnp.float64),
    )

    # Free large temporaries ASAP
    del knt_all, ijk_all

    scale = float(size ** 3)
    N = int(knt.shape[0])
    E = int(geom.conn.shape[0])
    meta: Dict[str, Any] = {
        "p2_path": p2_path,
        "size": float(size),
        "scale_size_cubed": scale,
        "K": float(K),
        "KL": float(KL),
        "auto_layers": bool(auto_layers),
        "nodes": N,
        "elements": E,
        # Optionally expose the scaling for diagnostics:
        "volume_scalefactor": float(volume_scalefactor),
        "body_volume_unscaled": float(body_vol_sum),
    }
    return knt, geom, meta
