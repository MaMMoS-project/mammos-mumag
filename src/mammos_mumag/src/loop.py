from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List
from pathlib import Path
import argparse
import configparser
import re
import jax
import jax.numpy as jnp
import sys
import time
from magnetostatics import (
    prepare_shells_and_geom,
    p2_path_for_mesh,
    read_mesh_params_from_p2,
    MU0,  # keep MU0 import
)
from geom import TetGeom
from energies import (
    brown_energy_and_grad_from_m,
    brown_energy_and_grad_from_scalar_potential,
    magnetic_volume,
)
from optimize import (
    prepare_core_amg,
    prepare_core_amg_scalar,  # <-- scalar AMG prep
    minimize_energy_lbfgs,
    minimize_energy_bb,
    make_uniform_u_raw,
    precompute_diag_tangent_from_geom,
    precompute_block_jacobi_3x3_from_geom,
)


# ------------------------------- Dataclasses ---------------------------------
@dataclass
class AmgPack:
    A_t: Tuple[Any, ...]
    P_t: Tuple[Any, ...]
    R_t: Tuple[Any, ...]
    Dinv_t: Tuple[jnp.ndarray, ...]
    L_c: Optional[jnp.ndarray] = None


@dataclass
class InitialState:
    mx: float
    my: float
    mz: float
    ini: int


@dataclass
class FieldSchedule:
    hstart: float
    hfinal: float
    hstep: float
    hx: float
    hy: float
    hz: float
    mstep: float
    mfinal: float


@dataclass
class MinimizerOptions:
    tol_fun: float
    tol_hmag_factor: float


@dataclass
class P2Config:
    path: str
    mesh_size: float
    initial: InitialState
    field: FieldSchedule
    minimizer: MinimizerOptions


@dataclass
class MaterialsPack:
    Ms_lookup: jnp.ndarray
    A_lookup_exchange: jnp.ndarray
    K1_lookup: jnp.ndarray
    k_easy_e: jnp.ndarray


@dataclass
class EnergyReport:
    S_brown: float
    S_brown_scaled: float
    E_classical: float
    E_classical_scaled: float
    Nz: float


# ---------------------------------- Utils ------------------------------------
_FLOAT_RX = re.compile(
    r"""
 \[\-\+\]?\(?\d+\(?\.\d*\)?|\.\d+\)?\([eE][\-\+]?\d+\)?
""",
    re.VERBOSE,
)


def _getfloat_relaxed(cfg, section, option, default=None):
    if not (cfg.has_section(section) and cfg.has_option(section, option)):
        return default
    try:
        return cfg.getfloat(section, option)
    except Exception:
        try:
            raw = cfg.get(section, option, fallback="")
            m = _FLOAT_RX.search(raw)
            return float(m.group(0)) if m else default
        except Exception:
            return default


_INT_RX = re.compile(
    r"""
    [\-\+]?          # optional sign
    \d+              # one or more digits
""",
    re.VERBOSE,
)


def _getint_relaxed(cfg, section, option, default=None):
    """
    Read an integer from a ConfigParser with relaxed parsing.

    Behavior:
    - If 'section' or 'option' is missing -> return default.
    - Try cfg.getint(section, option).
    - If that fails, scan the raw string and extract the first signed
      decimal integer substring; return int(value) if found.
    - On failure, return default.

    Parameters
    ----------
    cfg : configparser.ConfigParser
    section : str
    option : str
    default : Optional[int]

    Returns
    -------
    int or default
    """
    if not (cfg.has_section(section) and cfg.has_option(section, option)):
        return default
    try:
        return cfg.getint(section, option)
    except Exception:
        try:
            raw = cfg.get(section, option, fallback="")
            m = _INT_RX.search(raw)
            return int(m.group(0)) if m else default
        except Exception:
            return default


def _normalize(v: Tuple[float, float, float], eps=1e-30):
    x, y, z = v
    n = (x * x + y * y + z * z) ** 0.5
    return (0.0, 0.0, 1.0, 1.0) if n < eps else (x / n, y / n, z / n, n)


def _infer_default_krn_path(mesh_arg: str) -> Path:
    p = Path(mesh_arg)
    return p.with_suffix(".krn") if p.suffix else p.parent / (p.name + ".krn")


def _basename_from_mesh(mesh_arg: str) -> str:
    return str(Path(mesh_arg).with_suffix(""))


# ----------------------------- Steps 1–4 -------------------------------------
def step1_prepare(*, mesh: str, K=None, KL=None, auto_layers=True, verbose=False):
    p2p = p2_path_for_mesh(mesh)
    _, K_p2, KL_p2 = read_mesh_params_from_p2(p2p)
    K_eff = float(K if K is not None else (K_p2 if K_p2 is not None else 1.5))
    KL_eff = float(KL if KL is not None else (KL_p2 if KL_p2 is not None else 5.0))
    return prepare_shells_and_geom(
        mesh=mesh, K=K_eff, KL=KL_eff, auto_layers=auto_layers, verbose=verbose
    )


def step2_build_amg(
    geom: TetGeom, *, amg="sa", ms_mode: str = "A", gauge: float = 300.0
) -> AmgPack:
    if ms_mode == "A":
        A_t, P_t, R_t, Dinv_t, L_c = prepare_core_amg(geom, amg=amg, gauge=gauge)
    else:
        # Scalar: AMG on μ0*K, no gauge
        A_t, P_t, R_t, Dinv_t, L_c = prepare_core_amg_scalar(geom, amg=amg)
    return AmgPack(A_t=A_t, P_t=P_t, R_t=R_t, Dinv_t=Dinv_t, L_c=L_c)


def step3_read_p2(mesh: str) -> Optional[P2Config]:
    p2p = p2_path_for_mesh(mesh)
    p = Path(p2p)
    if not p.exists():
        return None
    cfg = configparser.ConfigParser()
    try:
        with p.open("r") as f:
            cfg.read_file(f)
    except Exception:
        return None

    size, _, _ = read_mesh_params_from_p2(p2p)
    mx = _getfloat_relaxed(cfg, "initial state", "mx", 0.0)
    my = _getfloat_relaxed(cfg, "initial state", "my", 0.0)
    mz = _getfloat_relaxed(cfg, "initial state", "mz", 1.0)
    ini = _getint_relaxed(cfg, "initial state", "ini", 0)

    hstart_T = _getfloat_relaxed(cfg, "field", "hstart", 0.0)
    hfinal_T = _getfloat_relaxed(cfg, "field", "hfinal", 0.0)
    hstep_T = _getfloat_relaxed(cfg, "field", "hstep", -0.002)
    hx = _getfloat_relaxed(cfg, "field", "hx", 0.0)
    hy = _getfloat_relaxed(cfg, "field", "hy", 0.0)
    hz = _getfloat_relaxed(cfg, "field", "hz", 1.0)
    hx_n, hy_n, hz_n, _ = _normalize((hx, hy, hz))

    mu0 = float(MU0)
    hstart = float(hstart_T) / mu0
    hfinal = float(hfinal_T) / mu0

    hstep = float(hstep_T) / mu0
    mstep = _getfloat_relaxed(cfg, "field", "mstep", 0.4)
    mfinal = _getfloat_relaxed(cfg, "field", "mfinal", -1.2)

    tol_fun = _getfloat_relaxed(cfg, "minimizer", "tol_fun", 1e-8)
    tol_hmag_factor = _getfloat_relaxed(cfg, "minimizer", "tol_hmag_factor", 1.0)

    init = InitialState(mx=float(mx), my=float(my), mz=float(mz), ini=int(ini))
    field = FieldSchedule(
        hstart=float(hstart),
        hfinal=float(hfinal),
        hstep=float(hstep),
        hx=float(hx_n),
        hy=float(hy_n),
        hz=float(hz_n),
        mstep=float(mstep),
        mfinal=float(mfinal),
    )
    minim = MinimizerOptions(
        tol_fun=float(tol_fun), tol_hmag_factor=float(tol_hmag_factor)
    )
    return P2Config(
        path=str(p), mesh_size=float(size), initial=init, field=field, minimizer=minim
    )


def step4_read_materials(
    *, krn_path: str, geom: TetGeom, mesh_size: float
) -> MaterialsPack:
    p = Path(krn_path)
    if not p.exists():
        raise FileNotFoundError(f".krn not found: {krn_path}")
    lines = p.read_text().splitlines()
    recs = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 6:
            continue
        theta, phi, K1, _unused, Js, A = map(float, parts[:6])
        recs.append((theta, phi, K1, Js, A))
    if not recs:
        raise ValueError(".krn has no valid data lines")

    G = int(jnp.max(geom.mat_id).item())
    if G < 1:
        raise ValueError("geom.mat_id must contain positive (1-based) IDs")
    M_needed = G - 1
    if len(recs) < M_needed:
        raise ValueError(f".krn provides {len(recs)} line(s); mesh needs {M_needed}")
    if len(recs) > M_needed:
        print(f"[warn] .krn has {len(recs)} lines; using first {M_needed}")
        recs = recs[:M_needed]

    inv_size2 = 1.0 / (mesh_size * mesh_size)
    Ms_list, Aex_list, K1_list, easy_rows = [], [], [], []
    for theta, phi, K1, Js, A in recs:
        Ms_list.append(float(Js) / float(MU0))
        Aex_list.append(float(A) * inv_size2)
        K1_list.append(float(K1))
        st, ct = jnp.sin(theta), jnp.cos(theta)
        sp, cp = jnp.sin(phi), jnp.cos(phi)
        easy_rows.append(jnp.asarray([st * cp, st * sp, ct], dtype=jnp.float64))

    # air material appended
    Ms_list.append(0.0)
    Aex_list.append(0.0)
    K1_list.append(0.0)

    Ms_lookup = jnp.asarray(Ms_list, dtype=jnp.float64)
    A_lookup_exchange = jnp.asarray(Aex_list, dtype=jnp.float64)
    K1_lookup = jnp.asarray(K1_list, dtype=jnp.float64)

    E = int(geom.conn.shape[0])
    EasyLUT = jnp.zeros((G + 1, 3), dtype=jnp.float64)
    for g in range(1, G):
        EasyLUT = EasyLUT.at[g].set(easy_rows[g - 1])
    k_easy_e = EasyLUT[geom.mat_id]

    return MaterialsPack(Ms_lookup, A_lookup_exchange, K1_lookup, k_easy_e)


# ----------------------------- Step 5/6 helpers ------------------------------
def compute_B_H_from_A(*, geom: TetGeom, A_nodes, m_nodes, Ms_lookup):
    conn = geom.conn
    grad_phi = geom.grad_phi
    mat_id = geom.mat_id
    A_e4 = A_nodes[conn]
    G_e4 = grad_phi
    B_e4 = jnp.cross(G_e4, A_e4)
    B_e = jnp.sum(B_e4, axis=1)
    Ms_e = jnp.take(Ms_lookup, (mat_id - 1))
    m_e = jnp.mean(m_nodes[conn], axis=1)
    M_e = Ms_e[:, None] * m_e
    H_e = (1.0 / float(MU0)) * B_e - M_e
    return M_e, B_e, H_e


def compute_B_H_from_U(*, geom: TetGeom, U_nodes, m_nodes, Ms_lookup):
    """
    Scalar potential route:
      H_e = -grad(U)_e  (constant per tet);  M_e = Ms_e * mean(m_nodes at tet);
      B_e = μ0 * (H_e + M_e)
    """
    conn = geom.conn
    grad_phi = geom.grad_phi
    mat_id = geom.mat_id
    U_e4 = U_nodes[conn]  # (E,4)
    # gradU_e[k] = sum_alpha U_e4[alpha] * grad_phi[alpha,k]
    gradU_e = jnp.einsum("ea,eak->ek", U_e4, grad_phi)  # (E,3)
    H_e = -gradU_e
    Ms_e = jnp.take(Ms_lookup, (mat_id - 1))
    m_e = jnp.mean(m_nodes[conn], axis=1)
    M_e = Ms_e[:, None] * m_e
    B_e = jnp.asarray(MU0, dtype=H_e.dtype) * (H_e + M_e)
    return M_e, B_e, H_e


# ----------------------------- Step 5: initial E -----------------------------
def check_unit_vectors(m: jnp.ndarray, tol: float = 1e-12) -> bool:
    """
    Check if all rows in m have norm ≈ 1 within a tolerance.

    Parameters
    ----------
    m : jnp.ndarray
        Magnetization array of shape (N, 3).
    tol : float
        Allowed deviation from 1.0 for the norm.

    Returns
    -------
    bool
        True if all norms are within [1 - tol, 1 + tol], False otherwise.
    """
    norms = jnp.linalg.norm(m, axis=1)
    return bool(jnp.all(jnp.abs(norms - 1.0) <= tol))


def make_vortex_yz(
    knt: jnp.ndarray,
    *,
    center: Optional[Sequence[float]] = None,
    core_radius_frac: float = 0.001,
    core_amp: float = 1.0,
    chirality: int = +1,  # +1 or -1: rotation sense in the y–z plane
    polarity: int = +1,  # +1 or -1: +x or -x vortex core
) -> jnp.ndarray:
    """
    Create a vortex in the y–z plane (circulation around x) with a small core along x.

    Parameters
    ----------
    knt : jnp.ndarray
        Node coordinates of shape (N, 3).
    center : (3,) sequence or None
        Vortex center. If None, the geometric center of knt's bounding box is used.
    core_radius_frac : float
        Core radius as a fraction of the characteristic y–z span (0..1).
        The actual radius r0 = core_radius_frac * max(Δy, Δz).
    core_amp : float
        Amplitude of the x-directed core term before normalization.
    chirality : {+1, -1}
        +1 gives e_phi = (0, -(z-cz)/rho, (y-cy)/rho); -1 flips the in-plane swirl.
    polarity : {+1, -1}
        +1 sets the core along +x; -1 along -x.

    Returns
    -------
    m : jnp.ndarray
        Unit magnetization field of shape (N, 3) representing a y–z-plane vortex.
    """
    knt = jnp.asarray(knt)
    dtype = knt.dtype

    # --- center of vortex
    if center is None:
        mins = jnp.min(knt, axis=0)
        maxs = jnp.max(knt, axis=0)
        c = 0.5 * (mins + maxs)
    else:
        c = jnp.asarray(center, dtype=dtype)

    # --- coordinates relative to center; only y, z matter for ρ and e_phi
    d = knt - c  # (N, 3)
    dy = d[:, 1]
    dz = d[:, 2]

    # --- radial distance in the y–z plane and core length scale
    rho = jnp.sqrt(dy * dy + dz * dz)
    # characteristic span in y–z plane
    min_xyz = jnp.min(knt, axis=0)
    max_xyz = jnp.max(knt, axis=0)
    dy_span = max_xyz[1] - min_xyz[1]
    dz_span = max_xyz[2] - min_xyz[2]
    r0 = core_radius_frac * jnp.maximum(dy_span, dz_span)

    # --- azimuthal unit vector around x (in-plane y–z circulation)
    #     e_phi = (0, -dz/rho, dy/rho) with chirality ±1
    eps = jnp.asarray(1e-30, dtype=dtype)
    inv_rho = 1.0 / jnp.maximum(rho, eps)
    ephi_x = jnp.zeros_like(rho)
    ephi_y = chirality * (-dz * inv_rho)
    ephi_z = chirality * (dy * inv_rho)

    # At the core (rho ~ 0), make e_phi = 0 explicitly to avoid tiny noise
    mask_core = rho < jnp.asarray(1e-12, dtype=rho.dtype)
    ephi_y = jnp.where(mask_core, 0.0, ephi_y)
    ephi_z = jnp.where(mask_core, 0.0, ephi_z)

    # --- x-directed vortex core (Gaussian)
    #     m_x_core = polarity * core_amp * exp(-(rho/r0)^2)
    # Handle r0==0 (degenerate geometry) gracefully.
    inv_r0 = jnp.where(r0 > 0, 1.0 / r0, 0.0)
    mx_core = polarity * core_amp * jnp.exp(-((rho * inv_r0) ** 2))

    # --- Combine and normalize to unit length: v = e_phi + mx_core * ex
    v_x = mx_core
    v_y = ephi_y
    v_z = ephi_z
    v = jnp.stack([v_x, v_y, v_z], axis=1)

    nrm = jnp.linalg.norm(v, axis=1)
    nrm_safe = jnp.maximum(nrm, jnp.asarray(1e-12, dtype=dtype))
    m = v / nrm_safe[:, None]
    if check_unit_vectors(m):
        pass
    else:
        print("initialization error")
        sys.exit()
    return m


# Optional: JIT-compiled wrapper for speed
# make_vortex_yz_jit = jax.jit(make_vortex_yz, static_argnames=("center", "core_radius_frac", "core_amp", "chirality", "polarity"))


def step5_initial_energy_and_MS(
    *,
    knt,
    geom,
    amg: AmgPack,
    materials: MaterialsPack,
    p2cfg: P2Config,
    ini,
    ms_mode: str = "A",
    gauge: float = 300.0,
    tol=1e-3,
    maxiter=500,
    nu_pre=2,
    nu_post=2,
    omega=0.7,
    coarse_iters=8,
    coarse_omega=0.7,
    Nz: float = 1.0 / 3.0,
):
    N = int(knt.shape[0])

    if ini:  # set ini from cli
        if ini == "uniform":
            mx, my, mz = 0.0, 0.0, 1.0
            m0 = jnp.tile(jnp.asarray([mx, my, mz], dtype=jnp.float64), (N, 1))
        elif ini == "vortex":
            m0 = make_vortex_yz(knt)
        else:
            print("unknown initial state")
            sys.exit()
    else:
        mx, my, mz = (
            float(p2cfg.initial.mx),
            float(p2cfg.initial.my),
            float(p2cfg.initial.mz),
        )
        mx, my, mz, _ = _normalize((mx, my, mz))
        m0 = jnp.tile(jnp.asarray([mx, my, mz], dtype=jnp.float64), (N, 1))

    if ms_mode == "A":
        from magnetostatics import _solve_A_jax_cg_compMG_core_jit

        x0_flat = jnp.zeros((3 * N,), dtype=jnp.float64)
        A0, *_ = _solve_A_jax_cg_compMG_core_jit(
            amg.A_t,
            amg.P_t,
            amg.R_t,
            amg.Dinv_t,
            amg.L_c,
            x0_flat,
            m0,
            geom,
            materials.Ms_lookup,
            tol,
            maxiter,
            gauge=gauge,
            nu_pre=nu_pre,
            nu_post=nu_post,
            omega=omega,
            coarse_iters=coarse_iters,
            coarse_omega=coarse_omega,
        )
        S, _, _ = brown_energy_and_grad_from_m(m0, A0, geom, materials.Ms_lookup)
        aux0 = A0
    else:
        from magnetostatics import _solve_U_jax_cg_compMG_core_jit

        x0_u = jnp.zeros((N,), dtype=jnp.float64)
        U0, *_ = _solve_U_jax_cg_compMG_core_jit(
            amg.A_t,
            amg.P_t,
            amg.R_t,
            amg.Dinv_t,
            amg.L_c,
            x0_u,
            m0,
            geom,
            materials.Ms_lookup,
            tol,
            maxiter,
            nu_pre=nu_pre,
            nu_post=nu_post,
            omega=omega,
            coarse_iters=coarse_iters,
            coarse_omega=coarse_omega,
        )
        S, _, _ = brown_energy_and_grad_from_scalar_potential(
            m0, U0, geom, materials.Ms_lookup
        )
        aux0 = U0

    Ms_e = jnp.take(materials.Ms_lookup, (geom.mat_id - 1))
    E_class = 0.5 * float(MU0) * float(Nz) * float(jnp.sum((Ms_e**2) * geom.volume))

    size_cubed = float(p2cfg.mesh_size**3)
    vol_scale = float(geom.volume_scalefactor)

    # NOTE: Energy density printed elsewhere = E_norm * E_ref
    report = EnergyReport(
        S_brown=float(S),
        S_brown_scaled=float(S) * size_cubed / vol_scale,
        E_classical=E_class,
        E_classical_scaled=E_class * size_cubed / vol_scale,
        Nz=float(Nz),
    )
    return m0, aux0, report


# ----------------------------- Step 6: fields to VTU -------------------------
def write_vtu_MHB(
    *, basename, knt, geom, mat_id, M_elems, H_elems, B_elems, index=None
):
    try:
        import meshio
    except Exception as exc:
        raise RuntimeError("meshio required: pip install meshio") from exc
    import numpy as np

    mu0 = float(MU0)
    M_T = np.asarray(M_elems * mu0, dtype=np.float64)
    H_T = np.asarray(H_elems * mu0, dtype=np.float64)
    B_T = np.asarray(B_elems, dtype=np.float64)
    points = np.asarray(knt, dtype=np.float64)
    cells = [("tetra", np.asarray(geom.conn, dtype=np.int32))]
    cell_data = {
        "mat_id": [np.asarray(mat_id, dtype=np.int32)],
        "M": [M_T],
        "B": [B_T],
        "H": [H_T],
    }
    mesh = meshio.Mesh(points=points, cells=cells, cell_data=cell_data)
    out_path = (
        f"{basename}.0000.vtu" if index is None else f"{basename}.{index:04d}.vtu"
    )
    mesh.write(out_path)
    return out_path


def save_state_with_index(
    *, basename: str, index: int, u_raw, aux_star, ms_mode: str
) -> str:
    """
    Save current state (u_raw and aux_star) to an .npz named with the VTU index.
    File name: {basename}.{index:04d}.state.npz

    Parameters
    ----------
    basename : str
        Base path used for VTU files.
    index : int
        VTU index (zero-padded to 4 digits).
    u_raw : array_like (N,3)
        Current u_raw (your code assigns this to the normalized m-nodes).
    aux_star : array_like
        Current auxiliary magnetostatics variable:
        - ms_mode == "A": A_nodes with shape (N,3)
        - ms_mode == "U": U_nodes with shape (N,)
    ms_mode : {"A","U"}
        Magnetostatics formulation.

    Returns
    -------
    out_path : str
        Path of the written .npz.
    """
    import numpy as np

    mode_flag = 0 if ms_mode == "A" else 1
    out_path = f"{basename}.{index:04d}.state.npz"
    np.savez(
        out_path,
        u_raw=np.asarray(u_raw),
        aux_star=np.asarray(aux_star),
        ms_mode=np.array(mode_flag, dtype=np.int8),  # 0="A", 1="U"
        version=np.array(1, dtype=np.int32),
    )
    return out_path


def load_state_file(
    state_path: str, *, expected_N: int | None = None, ms_mode: str | None = None
):
    """
    Load a previously saved state (.npz) and return (u_raw, aux_star).

    Parameters
    ----------
    state_path : str
        Path to a file created by save_state_with_index().
    expected_N : int or None
        If provided, verify node count matches (N).
    ms_mode : {"A","U"} or None
        If provided, verify the saved file matches this formulation.

    Returns
    -------
    (u_raw, aux_star)
        JAX arrays ready to be used as initial state.
    """
    import numpy as np

    data = np.load(state_path, allow_pickle=False)

    # Extract mandatory arrays
    u_raw_np = data["u_raw"]
    aux_star_np = data["aux_star"]

    # Optional / legacy fields
    saved_mode_flag = int(data["ms_mode"]) if "ms_mode" in data.files else 0
    saved_mode = "A" if saved_mode_flag == 0 else "U"

    # Sanity checks
    if expected_N is not None and u_raw_np.shape[0] != int(expected_N):
        raise ValueError(
            f"State file node count mismatch: file N={u_raw_np.shape[0]} "
            f"!= expected N={expected_N}"
        )
    if ms_mode is not None and ms_mode != saved_mode:
        raise ValueError(f"State file ms_mode={saved_mode!r} != requested {ms_mode!r}")

    # Return as JAX arrays
    return jnp.asarray(u_raw_np), jnp.asarray(aux_star_np)


def _parse_int_or_none(s: str | None) -> int | None:
    """
    Return int(s) if s is a valid integer string (e.g., "7", "+7", "-3", "0032"),
    otherwise None. Passing None returns None.
    """
    if s is None:
        return None
    s = s.strip()
    if not s:
        return None
    try:
        return int(s, 10)
    except ValueError:
        return None


def make_resume_state_path(base, ini, initial):
    idx = _parse_int_or_none(ini)
    if idx:
        return f"{base}.{idx:04d}.state.npz"
    if initial.ini > 0:
        return f"{base}.{initial.ini:04d}.state.npz"
    return None


# ----------------------------- Step 7: sweep ---------------------------------
@jax.jit
def _mh_mxyz_device(
    *,
    geom: TetGeom,
    m_nodes: jnp.ndarray,
    Ms_lookup: jnp.ndarray,
    h_dir_unit: jnp.ndarray,
):
    Ms_e = jnp.take(Ms_lookup, (geom.mat_id - 1))
    m_e = jnp.mean(m_nodes[geom.conn], axis=1)
    M_e = Ms_e[:, None] * m_e
    Vmag = magnetic_volume(geom, Ms_lookup=Ms_lookup, Ms_tol=0.0)
    Vmag_safe = jnp.where(Vmag > 0.0, Vmag, 1.0)
    integral = jnp.sum(M_e * geom.volume[:, None], axis=0)
    Mx, My, Mz = integral / Vmag_safe
    Mx = jnp.where(Vmag > 0.0, Mx, 0.0)
    My = jnp.where(Vmag > 0.0, My, 0.0)
    Mz = jnp.where(Vmag > 0.0, Mz, 0.0)
    MH_num = jnp.sum(jnp.sum(M_e * h_dir_unit[None, :], axis=1) * geom.volume)
    MH = jnp.where(Vmag > 0.0, MH_num / Vmag_safe, 0.0)
    return MH, Mx, My, Mz


def _h_schedule(hstart, hfinal, hstep) -> List[float]:
    vals = []
    if hstep == 0.0:
        return [float(hstart)]
    if hstep > 0 and hstart > hfinal:
        hstep = -hstep
    if hstep < 0 and hstart < hfinal:
        hstep = -hstep
    x = hstart
    if hstep > 0:
        while x <= hfinal + 1e-15:
            vals.append(float(x))
            x += hstep
    else:
        while x >= hfinal - 1e-15:
            vals.append(float(x))
            x += hstep
    return vals


def _m_from_u_raw(u_raw: jnp.ndarray, eps: float = 1e-12):
    r = jnp.linalg.norm(u_raw, axis=1)
    r_safe = jnp.maximum(r, jnp.asarray(eps, u_raw.dtype))
    return u_raw / r_safe[:, None]


def _report_outer_params_host(
    *,
    solver: str,  # "lbfgs" or "bb"
    # --- shared across both drivers ---
    ms_mode: str,
    gauge: float,
    # Core A/U-solver tuning:
    tol: float,
    maxiter: int,
    nu_pre: int,
    nu_post: int,
    omega: float,
    coarse_iters: int,
    coarse_omega: float,
    # Mapping (fixed inside step7):
    eps_norm: float = 1e-12,
    # Outer stopping:
    outer_max_iter: int,
    grad_tol: float,
    # --- LBFGS-only knobs (set to None when solver=="bb") ---
    history_size: int | None = None,
    debug_lbfgs: bool | None = None,
    H0_mode: str | None = None,
    ls_init_mode: str | None = None,
    ls_init_stepsize: float | None = None,
    ls_max_stepsize: float | None = None,
    ls_increase_factor: float | None = None,
    ls_c1: float | None = None,
    ls_c2: float | None = None,
    ls_decrease: float | None = None,
    ls_increase: float | None = None,
    diag_u=None,  # None | (N,) | (N,3,3)
    # --- BB-only knobs (set to None when solver=="lbfgs") ---
    bb_variant: str | None = None,
    debug_bb: bool | None = None,
    bb2_precond: bool | None = None,
    bb_init_steps: int | None = None,
    precond_mode: str | None = None,  # "none"|"diag"|"block_jacobi"
):
    """Human-readable outer solver configuration (host-side, outside JIT)."""
    print("\n----------------------------------------------------------------------")
    head = f"[Outer Params :: {solver.upper()}] magnetostatics={('U (scalar)' if ms_mode == 'U' else 'A (vector)')}"
    print(head)
    print(
        f" core-solver: tol={tol:.3e} maxiter={maxiter} "
        f"nu_pre={nu_pre} nu_post={nu_post} omega={omega:.3f} "
        f"coarse_iters={coarse_iters} coarse_omega={coarse_omega:.3f} gauge={gauge:.6g}"
    )
    print(
        f" mapping/stop: eps_norm={eps_norm:.3e} outer_max_iter={outer_max_iter} tol_fun={grad_tol:.3e}"
    )

    if solver.lower() == "lbfgs":
        print(" LBFGS:")
        print(f"  history={history_size} debug={bool(debug_lbfgs)} H0_mode={H0_mode}")
        print(
            f"  line-search: init_mode={ls_init_mode} init_stepsize={ls_init_stepsize:.3e} "
            f"max_stepsize={ls_max_stepsize:.3e} increase_factor={ls_increase_factor}"
        )
        print(f"  decrease={ls_decrease} increase={ls_increase}  c1={ls_c1} c2={ls_c2}")
        if diag_u is None:
            print("  preconditioner payload: diag_u=None")
        else:
            # summarize diag_u payload without dumping data
            try:
                sh = getattr(diag_u, "shape", None)
                dt = str(getattr(diag_u, "dtype", None))
                print(f"  preconditioner payload: diag_u shape={sh} dtype={dt}")
            except Exception:
                print("  preconditioner payload: diag_u=<unavailable>")
    else:
        print(" BB:")
        print(
            f"  variant={bb_variant} debug={bool(debug_bb)}  "
            f"bb_init_steps={bb_init_steps}"
        )
        print(
            f"  line-search: init_stepsize={ls_init_stepsize:.3e} "
            f"max_stepsize={ls_max_stepsize:.3e}"
        )
        print(f"  decrease={ls_decrease}  c1={ls_c1:.3e}")
    print("----------------------------------------------------------------------\n")


def step7_demag_sweep(
    *,
    basename,
    knt,
    geom,
    amg: AmgPack,
    materials: MaterialsPack,
    p2cfg: P2Config,
    ini,
    energies0: EnergyReport,
    ms_mode: str = "A",
    gauge: float = 300.0,
    a_tol=1e-3,
    a_maxiter=500,
    a_nu_pre=2,
    a_nu_post=2,
    a_omega=0.7,
    a_coarse_iters=8,
    a_coarse_omega=0.7,
    lbfgs_history=5,
    lbfgs_it=800,
    grad_tol=1e-3,
    debug_lbfgs=False,
    h0_mode="diag",
    h0_damping=1e-10,
    ls_init="current",
    ls_init_stepsize=1.0,
    ls_pass_init=False,
    ls_max_stepsize=1.0,
    ls_increase_factor=1.5,
    ls_c1=1e-4,
    ls_c2=0.9,
    ls_decrease=0.5,
    ls_increase=2.0,
    solver="lbfgs",
    bb_variant="alt",
    bb2_precond=True,
    bb_init_steps=2,
):
    t0 = time.perf_counter()
    total_fun_evals = 0
    total_outer_iters = 0
    method_used = solver

    N = int(knt.shape[0])
    field = p2cfg.field
    h_dir_unit = jnp.asarray([field.hx, field.hy, field.hz], dtype=jnp.float64)
    E_ref = jnp.asarray(energies0.E_classical, dtype=jnp.float64)

    alpha_prev = ls_init_stepsize

    # Diagonal / block-Jacobi seeds use CLI damping
    if h0_mode == "diag":
        diag_u = precompute_diag_tangent_from_geom(
            geom,
            A_lookup_exchange=materials.A_lookup_exchange,
            K1_lookup=materials.K1_lookup,
            E_ref=E_ref,
            mu=float(h0_damping),
        )
    elif h0_mode == "block_jacobi":
        diag_u = precompute_block_jacobi_3x3_from_geom(
            geom,
            A_lookup_exchange=materials.A_lookup_exchange,
            K1_lookup=materials.K1_lookup,
            k_easy_e=materials.k_easy_e,
            E_ref=E_ref,
            mu=float(h0_damping),
            return_inverse=True,
        )
    else:
        diag_u = None

    resume_state_path = make_resume_state_path(basename, ini, p2cfg.initial)

    if resume_state_path:
        u_raw_loaded, aux_loaded = load_state_file(
            resume_state_path, expected_N=N, ms_mode=ms_mode
        )
        u_raw = u_raw_loaded
        aux_prev = aux_loaded

    else:
        if ini:  # set ini from cli
            if ini == "uniform":
                mx, my, mz = 0.0, 0.0, 1.0
                # InitialState dataclass requires an 'ini' field; when constructing
                # a temporary InitialState for uniform initialization (CLI), set
                # ini=0 as a sensible default.
                u_raw = make_uniform_u_raw(
                    N, InitialState(mx, my, mz, ini=0), scale=1.0
                )
            elif ini == "vortex":
                u_raw = make_vortex_yz(knt)
            else:
                print("unknown initial state")
                sys.exit()
        else:
            u_raw = make_uniform_u_raw(N, init=p2cfg.initial, scale=1.0)

        aux_prev = (
            jnp.zeros((knt.shape[0], 3), dtype=jnp.float64)
            if ms_mode == "A"
            else jnp.zeros((knt.shape[0],), dtype=jnp.float64)
        )

    h_vals = _h_schedule(field.hstart, field.hfinal, field.hstep)
    dat_path = f"{basename}.dat"
    vtu_index = 0
    # If we resume from a numbered state, continue VTU numbering from there.
    # (We increment before writing, so starting from the resume index will produce
    # resume_index + 1 for the next output.)
    resume_index = None
    if resume_state_path:
        _m = re.search(r"\.(\d{4})\.state\.npz$", str(resume_state_path))
        if _m:
            try:
                resume_index = int(_m.group(1))
                vtu_index = resume_index
            except Exception:
                pass
    last_MH_mu0 = None

    # --- print once before the field loop ---
    if solver == "lbfgs":
        _report_outer_params_host(
            solver="lbfgs",
            ms_mode=ms_mode,
            gauge=(gauge if ms_mode == "A" else 0.0),
            tol=a_tol,
            maxiter=a_maxiter,
            nu_pre=a_nu_pre,
            nu_post=a_nu_post,
            omega=a_omega,
            coarse_iters=a_coarse_iters,
            coarse_omega=a_coarse_omega,
            eps_norm=1e-12,
            outer_max_iter=lbfgs_it,
            grad_tol=grad_tol,
            history_size=lbfgs_history,
            debug_lbfgs=debug_lbfgs,
            H0_mode=h0_mode,
            ls_init_mode=ls_init,
            ls_init_stepsize=ls_init_stepsize,
            ls_max_stepsize=ls_max_stepsize,
            ls_increase_factor=ls_increase_factor,
            ls_c1=ls_c1,
            ls_c2=ls_c2,
            ls_decrease=ls_decrease,
            ls_increase=ls_increase,
            diag_u=diag_u,
        )
    else:
        _report_outer_params_host(
            solver="bb",
            ms_mode=ms_mode,
            gauge=(gauge if ms_mode == "A" else 0.0),
            tol=a_tol,
            maxiter=a_maxiter,
            nu_pre=a_nu_pre,
            nu_post=a_nu_post,
            omega=a_omega,
            coarse_iters=a_coarse_iters,
            coarse_omega=a_coarse_omega,
            eps_norm=1e-12,
            outer_max_iter=lbfgs_it,
            grad_tol=grad_tol,  # 'lbfgs_it' reused as outer_max_iter for BB
            bb_variant=bb_variant,
            debug_bb=debug_lbfgs,  # same debug flag reused
            bb_init_steps=bb_init_steps,
            ls_init_stepsize=ls_init_stepsize,
            ls_max_stepsize=ls_max_stepsize,
            ls_c1=ls_c1,
            ls_decrease=ls_decrease,
        )

    # header for output
    with open(dat_path, "w", encoding="utf-8") as f:
        f.write(
            f"#{'vtu':>3} "
            f"{'mu0 Hext(T)':>13} {'J.h(T)':>12} "
            f"{'Jx(T)':>10} {'Jy(T)':>10} {'Jz(T)':>10} "
            f"{'e(J/m3)':>10} {'e_ms(J/m3)':>10} {'e_ex(J/m3)':>10} {'e_an(J/m3)':>10} {'e_ze(J/m3)':>10}\n"
        )

    print(
        f"# {'mu0 Hext(T)':>13} {'J.h(T)':>13} "
        f"{'Jx(T)':>11} {'Jy(T)':>11} {'Jz(T)':>11} "
        f"{'e(J/m3)':>10} {'e_ms(J/m3)':>10} {'e_ex(J/m3)':>10} {'e_an(J/rm3)':>10} {'e_ze(J/m3)':>10}"
    )

    for hmag in h_vals:
        H_ext = jnp.asarray(
            [field.hx * hmag, field.hy * hmag, field.hz * hmag], dtype=jnp.float64
        )

        if solver == "lbfgs":
            E_norm, u_star, aux_star, parts_norm, alpha_out, metrics = (
                minimize_energy_lbfgs(
                    initial_u_raw=u_raw,
                    initial_A=aux_prev,
                    A_t=amg.A_t,
                    P_t=amg.P_t,
                    R_t=amg.R_t,
                    Dinv_t=amg.Dinv_t,
                    L_c=amg.L_c,
                    conn=geom.conn,
                    grad_phi=geom.grad_phi,
                    volume=geom.volume,
                    mat_id=geom.mat_id,
                    Ms_lookup=materials.Ms_lookup,
                    A_lookup_exchange=materials.A_lookup_exchange,
                    K1_lookup=materials.K1_lookup,
                    k_easy_e=materials.k_easy_e,
                    H_ext=H_ext,
                    E_ref=E_ref,
                    tol=a_tol,
                    maxiter=a_maxiter,
                    nu_pre=a_nu_pre,
                    nu_post=a_nu_post,
                    omega=a_omega,
                    coarse_iters=a_coarse_iters,
                    coarse_omega=a_coarse_omega,
                    eps_norm=1e-12,
                    gauge=(gauge if ms_mode == "A" else 0.0),
                    ms_mode=ms_mode,
                    history_size=lbfgs_history,
                    outer_max_iter=lbfgs_it,
                    grad_tol=grad_tol,
                    debug_lbfgs=debug_lbfgs,
                    H0_mode=h0_mode,
                    ls_init_mode=ls_init,
                    ls_init_stepsize=alpha_prev,
                    ls_max_stepsize=ls_max_stepsize,
                    ls_increase_factor=ls_increase_factor,
                    ls_c1=ls_c1,
                    ls_c2=ls_c2,
                    ls_decrease=ls_decrease,
                    ls_increase=ls_increase,
                    diag_u=diag_u,
                )
            )
        else:
            E_norm, u_star, aux_star, parts_norm, alpha_out, metrics = (
                minimize_energy_bb(
                    initial_u_raw=u_raw,
                    initial_A=aux_prev,
                    A_t=amg.A_t,
                    P_t=amg.P_t,
                    R_t=amg.R_t,
                    Dinv_t=amg.Dinv_t,
                    L_c=amg.L_c,
                    conn=geom.conn,
                    grad_phi=geom.grad_phi,
                    volume=geom.volume,
                    mat_id=geom.mat_id,
                    Ms_lookup=materials.Ms_lookup,
                    A_lookup_exchange=materials.A_lookup_exchange,
                    K1_lookup=materials.K1_lookup,
                    k_easy_e=materials.k_easy_e,
                    H_ext=H_ext,
                    E_ref=E_ref,
                    tol=a_tol,
                    maxiter=a_maxiter,
                    nu_pre=a_nu_pre,
                    nu_post=a_nu_post,
                    omega=a_omega,
                    coarse_iters=a_coarse_iters,
                    coarse_omega=a_coarse_omega,
                    eps_norm=1e-12,
                    gauge=(gauge if ms_mode == "A" else 0.0),
                    ms_mode=ms_mode,
                    outer_max_iter=lbfgs_it,  # reuse same CLI knob for iterations
                    grad_tol=grad_tol,  # reuse tol from CLI
                    bb_variant=bb_variant,  # or "bb1"/"bb2" if you prefer
                    debug_bb=debug_lbfgs,  # reuse debug flag
                    ls_init_stepsize=alpha_prev,
                    ls_max_stepsize=ls_max_stepsize,
                    ls_c1=ls_c1,
                    ls_decrease=ls_decrease,
                    bb_init_steps=bb_init_steps,
                )
            )

        if ls_pass_init:
            alpha_prev = float(alpha_out)

        total_outer_iters += int(metrics[0])
        total_fun_evals += int(metrics[1])
        m_nodes = _m_from_u_raw(u_star)
        u_raw = m_nodes

        MH, Mx, My, Mz = _mh_mxyz_device(
            geom=geom,
            m_nodes=m_nodes,
            Ms_lookup=materials.Ms_lookup,
            h_dir_unit=h_dir_unit,
        )
        MH_mu0 = float(MH * jnp.asarray(MU0, dtype=jnp.float64))

        # ---- Energy density [J/m^3] = E_norm * E_ref
        E_density = float(E_norm) * float(E_ref)

        S_norm = float(parts_norm[0])  # magnetostatic (Brown functional)
        Ex_norm = float(parts_norm[1])  # exchange
        An_norm = float(parts_norm[2])  # anisotropy
        Ze_norm = float(parts_norm[3])  # Zeeman

        S_density = S_norm * float(E_ref)
        Ex_density = Ex_norm * float(E_ref)
        An_density = An_norm * float(E_ref)
        Ze_density = Ze_norm * float(E_ref)

        write_now = (
            (last_MH_mu0 is None)
            or (abs(MH_mu0 - last_MH_mu0) >= field.mstep)
            or (hmag == h_vals[-1])
            or (abs(hmag) < 1e-12)
        )
        vtu_written_id = 0
        if write_now:
            vtu_index += 1
            if ms_mode == "A":
                M_e, B_e, H_e = compute_B_H_from_A(
                    geom=geom,
                    A_nodes=aux_star,
                    m_nodes=m_nodes,
                    Ms_lookup=materials.Ms_lookup,
                )
            else:
                M_e, B_e, H_e = compute_B_H_from_U(
                    geom=geom,
                    U_nodes=aux_star,
                    m_nodes=m_nodes,
                    Ms_lookup=materials.Ms_lookup,
                )
            vtu_path = write_vtu_MHB(
                basename=basename,
                knt=knt,
                geom=geom,
                mat_id=geom.mat_id,
                M_elems=M_e,
                H_elems=H_e,
                B_elems=B_e,
                index=vtu_index,
            )
            vtu_written_id = vtu_index
            last_MH_mu0 = MH_mu0
            # Persist state at the same index as the VTU
            try:
                _ = save_state_with_index(
                    basename=basename,
                    index=vtu_index,
                    u_raw=u_raw,  # normalized m-nodes (the code sets u_raw = m_nodes)
                    aux_star=aux_star,  # A_nodes (A) or U_nodes (U)
                    ms_mode=ms_mode,
                )
                # Optional: print confirmation
                # print(f"[Step 7] State written -> {_}")
            except Exception as _exc:
                print(
                    f"[warn] Could not write state file for index {vtu_index}: {_exc}"
                )

        # ---- Console output

        print(
            f"  {float(hmag * MU0):13.6e} {float(MH * MU0):13.6e} "
            f"{float(Mx * MU0):11.4e} {float(My * MU0):11.4e} {float(Mz * MU0):11.4e} "
            f"{E_density:10.3e} {S_density:10.3e} {Ex_density:10.3e} {An_density:10.3e} {Ze_density:10.3e}"
        )

        # ---- .dat output
        with open(dat_path, "a", encoding="utf-8") as f:
            f.write(
                f"{vtu_written_id:03d} "
                f"{float(hmag * MU0):13.6e} {float(MH * MU0):12.5e} "
                f"{float(Mx * MU0):10.3e} {float(My * MU0):10.3e} {float(Mz * MU0):10.3e} "
                f"{E_density:10.3e} {S_density:10.3e} {Ex_density:10.3e} {An_density:10.3e} {Ze_density:10.3e}\n"
            )

        aux_prev = aux_star

    t_total = time.perf_counter() - t0
    print("\n[Report] Optimization summary")
    print(
        f" method: {method_used.upper()}  (magnetostatics: {'U (scalar)' if ms_mode == 'U' else 'A (vector)'})"
    )
    print(f" total wall time: {t_total:.3f} s")
    print(f" total function evaluations: {total_fun_evals}")
    print(f" total nonlinear iterations ({method_used}): {total_outer_iters}")


# ---------------------------------- CLI --------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Micromagnetics (Steps 1–7) with two-loop L-BFGS."
    )
    ap.add_argument("--mesh", required=True)
    ap.add_argument(
        "--size", type=float, default=None, help="overwrite mesh units in p2 file"
    )
    ap.add_argument(
        "--ini", default=None, help="Initial state (uniform, vortex, or number)"
    )
    ap.add_argument("--K", type=float, default=None)
    ap.add_argument("--KL", type=float, default=None)
    ap.add_argument("--amg", choices=["sa", "rs"], default="sa")

    # --- Gauge selection: manual or auto (A only) ---
    ap.add_argument(
        "--gauge",
        type=float,
        default=0.0,
        help=(
            "Tikhonov stabilization added to the curl–curl (A-operator): adds gauge*I. "
            "Use this to set the strength manually. "
            "Alternatively, provide --h and --L to compute gauge = theta*(h/L)^2 with theta=0.1."
        ),
    )

    # --- Magnetostatics formulation switch ---
    ap.add_argument(
        "--ms",
        "--magnetostatics",
        dest="ms_mode",
        choices=["A", "U", "vector", "scalar"],
        default="U",
        help="Magnetostatics formulation: 'A' (vector potential) or 'U' (scalar potential).",
    )

    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--print-p2", action="store_true")
    ap.add_argument("--krn", type=str, default=None)
    ap.add_argument("--no-auto-krn", action="store_true")
    ap.add_argument("--print-materials", action="store_true")
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--nz", type=float, default=1.0 / 3.0)
    ap.add_argument("--print-energy", action="store_true")
    ap.add_argument("--a-it", type=int, default=500)
    ap.add_argument("--a-nu-pre", type=int, default=2)
    ap.add_argument("--a-nu-post", type=int, default=2)
    ap.add_argument("--a-omega", type=float, default=0.7)
    ap.add_argument("--a-coarse-it", type=int, default=8)
    ap.add_argument("--a-coarse-omega", type=float, default=0.7)
    ap.add_argument("--no-vtu", action="store_true")
    ap.add_argument("--no-demag", action="store_true")
    ap.add_argument("--lbfgs-history", type=int, default=5)
    ap.add_argument("--lbfgs-it", type=int, default=800)
    ap.add_argument("--tol-fun", type=float, default=None)
    ap.add_argument("--tol-hmag-factor", type=float, default=None)
    ap.add_argument("--debug-lbfgs", action="store_true")
    ap.add_argument(
        "--h0",
        choices=["gamma", "bb", "bb1", "bb2", "diag", "block_jacobi", "identity"],
        default="block_jacobi",
        help="Two-loop seed H0.",
    )
    ap.add_argument(
        "--h0-damping",
        type=float,
        default=0.0,
        help="Small SPD damping used by 'diag' or 'block_jacobi' H0.",
    )
    ap.add_argument(
        "--ls-init",
        choices=["current", "max", "value", "increase", "alpha0"],
        default="current",
        help="Line-search initial step strategy (default: current)",
    )
    ap.add_argument(
        "--ls-init-stepsize",
        type=float,
        default=None,
        help="Initial step size (default: 1 for lbfgs, 100.0 for bb).",
    )
    ap.add_argument("--ls-pass-init", action="store_true")
    ap.add_argument(
        "--ls-max-stepsize", type=float, default=1.0e5, help="Maximum step size."
    )
    ap.add_argument(
        "--ls-increase-factor",
        type=float,
        default=1.5,
        help="Expansion factor when --ls-init=increase",
    )
    ap.add_argument(
        "--ls-kind",
        choices=[
            "default",
            "armijo",
            "goldstein",
            "wolfe",
            "strong-wolfe",
            "curvilinear-armijo",
        ],
        default="default",
        help=(
            "Line-search flavor for L-BFGS: 'default' keeps JAXopt's zoom Strong-Wolfe; "
            "otherwise use a backtracking line search with the chosen condition."
        ),
    )
    ap.add_argument(
        "--ls-c1",
        type=float,
        default=0.3,
        help="Line-search c1 (sufficient decrease) for backtracking LS.",
    )
    ap.add_argument(
        "--ls-c2",
        type=float,
        default=0.7,
        help="Line-search c2 (curvature) for backtracking Wolfe/Strong-Wolfe.",
    )
    ap.add_argument(
        "--ls-decrease",
        type=float,
        default=0.5,
        help="Backtracking decrease factor (shrink multiplier, in (0,1)).",
    )
    ap.add_argument(
        "--ls-increase",
        type=float,
        default=2.0,
        help="Linesearch increase factor (expandion multiplier, > 1).",
    )
    ap.add_argument(
        "--solver",
        choices=["lbfgs", "bb"],
        default="lbfgs",
        help="Outer minimizer: L-BFGS (two-loop) or Barzilai Borwein.",
    )
    ap.add_argument(
        "--bb-variant",
        choices=["alt", "bb1", "bb2"],
        default="alt",
        help="variant of Barzilai-Borwein method",
    )
    ap.add_argument("--bb2-precond", action="store_true")
    ap.add_argument(
        "--bb-init-steps",
        type=int,
        default=2,
        help="initial number of linesearch steps",
    )
    args = ap.parse_args()

    # Step 1
    knt, geom, meta = step1_prepare(
        mesh=args.mesh,
        K=(args.K if args.K is not None else 2.0),
        KL=(args.KL if args.KL is not None else 10.0),
        auto_layers=True,
        verbose=args.verbose,
    )
    N = int(knt.shape[0])
    E = int(geom.conn.shape[0])
    print("\n[Step 1] Mesh + Air Shells + Geometry")
    print(f" nodes: {N:,d}, elements: {E:,d}")
    print(
        f" shells: K={meta.get('K')}, KL={meta.get('KL')}, auto_layers={meta.get('auto_layers')}"
    )
    print(
        f" mesh size: {meta.get('size'):.6g} -> scale size^3 = {meta.get('scale_size_cubed'):.6g}"
    )
    print(
        f" volume_scalefactor = {meta.get('volume_scalefactor'):.6e} (1 / body_volume_unscaled={meta.get('body_volume_unscaled'):.6e})"
    )

    # Step 2: build AMG according to ms_mode
    ms_mode = "A" if args.ms_mode in ("A", "vector") else "U"
    amg_pack = step2_build_amg(
        geom,
        amg=args.amg,
        ms_mode=ms_mode,
        gauge=(float(args.gauge) if ms_mode == "A" else 0.0),
    )
    L_levels = len(amg_pack.A_t)
    print("\n[Step 2] AMG hierarchy")
    print(
        f" levels: {L_levels}, finest n={int(amg_pack.A_t[0].shape[0]):,d}, type={args.amg!r}"
    )
    if ms_mode == "A":
        print(f" gauge = {args.gauge:.6g}")
    if amg_pack.L_c is not None:
        print(f" coarsest Cholesky: shape={tuple(amg_pack.L_c.shape)}")

    # Step 3
    p2cfg = step3_read_p2(args.mesh)
    if args.size:
        mesh_size = args.size
    else:
        mesh_size = float(p2cfg.mesh_size) if p2cfg else float(meta.get("size", 1e-9))
    if args.print_p2:
        if p2cfg is None:
            print("\n[Step 3] .p2 file: not found")
        else:
            print("\n[Step 3] .p2 parameters")
            print(f" path: {p2cfg.path}")
            print(f" mesh.size = {p2cfg.mesh_size:.6g}")
            if args.ini:
                print(f" initial state: {args.ini}")
            else:
                ini = p2cfg.initial
                print(
                    f" initial state: mx={ini.mx:.6g}, my={ini.my:.6g}, mz={ini.mz:.6g}"
                )
            fld = p2cfg.field
            print(
                f" field sweep: hstart={fld.hstart:.6g}, hfinal={fld.hfinal:.6g}, hstep={fld.hstep:.6g} [A/m]"
            )
            print(
                f" field dir (unit): hx={fld.hx:.8f}, hy={fld.hy:.8f}, hz={fld.hz:.8f}"
            )
            print(
                f" write m(h) each Δ(±0M) = {fld.mstep:.6g} T; stop if m(h) < {fld.mfinal:.6g}"
            )
    if args.tol_fun:
        tol_fun = args.tol_fun
    else:
        tol_fun = p2cfg.minimizer.tol_fun if p2cfg else 1e-8
    if args.tol_hmag_factor:
        tol_hmag_factor = args.tol_hmag_factor
    else:
        tol_hmag_factor = p2cfg.minimizer.tol_hmag_factor if p2cfg else 1
    print(f" minimizer: tol_fun={tol_fun:.3e}, tol_hmag_factor={tol_hmag_factor:.6g}")
    a_tol = tol_hmag_factor * (tol_fun ** (1 / 3))

    # Step 4
    krn_path: Optional[Path] = None
    if args.krn is not None:
        krn_path = Path(args.krn)
    elif not args.no_auto_krn:
        candidate = _infer_default_krn_path(args.mesh)
        if candidate.exists():
            krn_path = candidate
        else:
            print(f"\n[info] --krn not provided and default not found: {candidate}")

    materials = None
    if krn_path is not None:
        materials = step4_read_materials(
            krn_path=str(krn_path), geom=geom, mesh_size=mesh_size
        )
    if args.print_materials and materials is not None:
        G = int(jnp.max(geom.mat_id).item())
        print("\n[Step 4] Materials")
        print(f" #groups={G} (1..{G - 1}=materials, {G}=air)")
        print(f" Ms_lookup [A/m]: {materials.Ms_lookup.tolist()}")
        print(
            f" A_lookup_exchange [J/m]: {materials.A_lookup_exchange.tolist()} (A/size^2)"
        )
        print(f" K1_lookup [J/m^3]: {materials.K1_lookup.tolist()}")

    # Step 5
    aux0 = None
    m0 = None
    energies0 = None
    if not args.no_energy and (p2cfg is not None) and (materials is not None):
        m0, aux0, energies0 = step5_initial_energy_and_MS(
            knt=knt,
            geom=geom,
            amg=amg_pack,
            materials=materials,
            p2cfg=p2cfg,
            ini=args.ini,
            ms_mode=ms_mode,
            gauge=(float(args.gauge) if ms_mode == "A" else 0.0),
            tol=a_tol,
            maxiter=args.a_it,
            nu_pre=args.a_nu_pre,
            nu_post=args.a_nu_post,
            omega=args.a_omega,
            coarse_iters=args.a_coarse_it,
            coarse_omega=args.a_coarse_omega,
            Nz=float(args.nz),
        )

    if args.print_energy and energies0 is not None:
        print("\n[Step 5] Initial energies")
        print(f" Brown (magnetostatic energy density) {energies0.S_brown:.6e} J/m3")
        print(f" Brown (magnetostatic energy) {energies0.S_brown_scaled:.6e} J")
        print(
            f" Magnetostatic energy density {energies0.E_classical:.6e} J/m3 (Nz={energies0.Nz:.6g})"
        )
        print(f" Magnetostatic energy {energies0.E_classical_scaled:.6e} J")
        print(
            f" Relative error {(energies0.S_brown - energies0.E_classical) / energies0.E_classical:.6e}"
        )

    # Step 6
    if (
        not args.no_vtu
        and (m0 is not None)
        and (aux0 is not None)
        and (materials is not None)
    ):
        if ms_mode == "A":
            M_e, B_e, H_e = compute_B_H_from_A(
                geom=geom, A_nodes=aux0, m_nodes=m0, Ms_lookup=materials.Ms_lookup
            )
        else:
            M_e, B_e, H_e = compute_B_H_from_U(
                geom=geom, U_nodes=aux0, m_nodes=m0, Ms_lookup=materials.Ms_lookup
            )
        base = _basename_from_mesh(args.mesh)
        vtu_path = write_vtu_MHB(
            basename=base,
            knt=knt,
            geom=geom,
            mat_id=geom.mat_id,
            M_elems=M_e,
            H_elems=H_e,
            B_elems=B_e,
            index=None,
        )
        print(f"\n[Step 6] VTU written -> {vtu_path}")
    elif args.no_vtu:
        print("\n[info] VTU writing disabled by --no-vtu")

    # Step 7
    if (
        (not args.no_demag)
        and (p2cfg is not None)
        and (materials is not None)
        and (energies0 is not None)
    ):
        print(
            "\n[Step 7] Demagnetization sweep: two-loop L-BFGS at each field step ..."
        )
        if args.ls_init_stepsize:
            ls_init_stepsize = args.ls_init_stepsize
        else:
            if args.solver == "bb":
                ls_init_stepsize = 100.0
            else:
                ls_init_stepsize = 1.0
        base = _basename_from_mesh(args.mesh)
        step7_demag_sweep(
            basename=base,
            knt=knt,
            geom=geom,
            amg=amg_pack,
            materials=materials,
            p2cfg=p2cfg,
            ini=args.ini,
            energies0=energies0,
            ms_mode=ms_mode,
            gauge=(float(args.gauge) if ms_mode == "A" else 0.0),
            a_tol=a_tol,
            a_maxiter=args.a_it,
            a_nu_pre=args.a_nu_pre,
            a_nu_post=args.a_nu_post,
            a_omega=args.a_omega,
            a_coarse_iters=args.a_coarse_it,
            a_coarse_omega=args.a_coarse_omega,
            lbfgs_history=args.lbfgs_history,
            lbfgs_it=args.lbfgs_it,
            grad_tol=tol_fun,
            debug_lbfgs=args.debug_lbfgs,
            h0_mode=args.h0,
            h0_damping=args.h0_damping,
            ls_init=args.ls_init,
            ls_init_stepsize=ls_init_stepsize,
            ls_pass_init=args.ls_pass_init,
            ls_max_stepsize=args.ls_max_stepsize,
            ls_increase_factor=args.ls_increase_factor,
            ls_c1=args.ls_c1,
            ls_c2=args.ls_c2,
            ls_decrease=args.ls_decrease,
            ls_increase=args.ls_increase,
            solver=args.solver,
            bb_variant=args.bb_variant,
            bb2_precond=args.bb2_precond,
            bb_init_steps=args.bb_init_steps,
        )
        print(f"[Step 7] Sweep finished. Appended results to {base}.dat")
    elif args.no_demag:
        print("\n[info] Demagnetization sweep disabled by --no-demag")


def main_cli():  # keep entrypoint name explicit
    main()


if __name__ == "__main__":
    main()
