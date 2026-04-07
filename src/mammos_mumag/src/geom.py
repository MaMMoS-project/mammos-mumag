import numpy as np
import jax
import jax.numpy as jnp
from typing import NamedTuple, Tuple, Optional
from functools import partial


class TetGeom(NamedTuple):
    conn: jnp.ndarray          # (E,4) int32
    grad_phi: jnp.ndarray      # (E,4,3) float64 -- constant ∇φ per element/vertex
    volume: jnp.ndarray        # (E,) float64 -- V_e (may be scaled)
    mat_id: jnp.ndarray        # (E,) int32 -- material group id per element
    volume_scalefactor: jnp.ndarray  # () float64 -- scale applied to volumes (default 1.0)


def _coeff_matrix(verts):
    """B = [[1, x1, y1, z1], ..., [1, x4, y4, z4]] (shape 4x4)."""
    ones = jnp.ones((4, 1), dtype=verts.dtype)
    return jnp.concatenate([ones, verts], axis=1)


def _tet_geom_from_verts(verts):
    """
    Compute per-element quantities from vertex coords (4,3):
    - grad_phi (4,3): rows are (b_i, c_i, d_i)
    - volume (scalar): det(B)/6 (Eq. 13)
    """
    B = _coeff_matrix(verts)  # (4,4)
    Binv = jnp.linalg.inv(B)  # columns are [a_i,b_i,c_i,d_i]
    detB = jnp.linalg.det(B)
    Ve = detB / 6.0          # Eq. (13)
    grads = Binv[1:, :].T    # (4,3) -> gradients (Eq. 18 via Eq. 8 coefficients)
    return grads, Ve


# vmap across elements
_vmap_geom = jax.vmap(_tet_geom_from_verts, in_axes=(0,), out_axes=(0, 0))


@partial(jax.jit, static_argnames=('one_based_indexing', 'fix_orientation'))
def precompute_linear_tet_geometry(
    knt,
    ijk,
    *,
    one_based_indexing: bool = False,
    fix_orientation: bool = True
) -> TetGeom:
    """
    Parameters
    ----------
    knt : (N,3) float64
        Node coordinates.
    ijk : (E,5) int32
        [n1, n2, n3, n4, mat_group]. Node ids may be 0- or 1-based.
    one_based_indexing : bool
        If True, node ids in ijk are converted to 0-based.
    fix_orientation : bool
        If True, ensures positive volume by swapping the last two nodes if needed.

    Returns
    -------
    TetGeom(conn, grad_phi, volume, mat_id, volume_scalefactor)
    """
    # Split connectivity and material ids
    conn_raw = ijk[:, :4]
    mat_id = ijk[:, 4].astype(jnp.int32)

    # Normalize indexing to 0-based if requested (static arg -> JIT-safe)
    conn = (conn_raw - 1) if one_based_indexing else conn_raw
    conn = conn.astype(jnp.int32)  # (E,4)

    # Gather element vertices: (E,4,3)
    verts = knt[conn]

    # Compute gradients of shape functions and signed volumes
    grad_phi, Ve = _vmap_geom(verts)  # grad_phi: (E,4,3), Ve: (E,)

    if fix_orientation:
        # Need to swap last two vertices where signed volume is negative
        neg = Ve < 0  # (E,)

        # JAX-safe vertex swap (no list indexing)
        def _swap_and_recompute(vs):
            # vs: (4,3)
            v2 = vs[2]
            v3 = vs[3]
            vs_swapped = vs.at[2].set(v3).at[3].set(v2)
            return _tet_geom_from_verts(vs_swapped)

        # For each element, either recompute with swapped verts or keep as-is
        grad_phi_fix, Ve_fix = jax.vmap(
            lambda vs, do: jax.lax.cond(do, _swap_and_recompute, _tet_geom_from_verts, vs),
            in_axes=(0, 0),
        )(verts, neg)

        # Select corrected (swapped) results where needed
        grad_phi = jnp.where(neg[:, None, None], grad_phi_fix, grad_phi)
        Ve = jnp.where(neg, Ve_fix, Ve)

        # Also keep connectivity consistent with the swap (swap local nodes 3<->4)
        swap_idx = jnp.array([0, 1, 3, 2], dtype=jnp.int32)  # swap the last two columns
        conn_swapped = conn[:, swap_idx]
        conn = jnp.where(neg[:, None], conn_swapped, conn)

    # Default: no scaling at this stage
    vol_scale = jnp.asarray(1.0, dtype=jnp.float64)
    return TetGeom(conn=conn, grad_phi=grad_phi, volume=Ve, mat_id=mat_id, volume_scalefactor=vol_scale)


# ------------------------------- New: high-level precompute function from in-memory arrays (NumPy or JAX)
def precompute_geometry_from_knt_ijk(
    knt_in: np.ndarray,
    ijk_in: np.ndarray,
    *,
    detect_one_based: Optional[bool] = None,
    fix_orientation: bool = True,
) -> Tuple[jnp.ndarray, TetGeom]:
    """
    Validate and precompute TetGeom directly from in-memory (knt, ijk).

    Parameters
    ----------
    knt_in : array-like, shape (N,3)
        Node coordinates (np.ndarray or jax array acceptable).
    ijk_in : array-like, shape (E,4) or (E,5)
        Connectivity. If (E,4), a default material group '1' is appended.
        Node ids may be 0- or 1-based; auto-detected by default.
    detect_one_based : Optional[bool]
        If True/False, force the 1-based detection outcome; if None, auto-detect.
    fix_orientation : bool
        If True, enforce positive tet volumes by swapping local nodes 3<->4 where needed.

    Returns
    -------
    knt : jax.numpy.ndarray (N,3) float64
        JAX array of node coordinates.
    geom : TetGeom
        Precomputed geometry (conn, grad_phi, volume, mat_id, volume_scalefactor).
    """
    # Convert to NumPy for shape checks (accept jax arrays too)
    knt_np = np.asarray(knt_in)
    ijk_np = np.asarray(ijk_in)
    if knt_np.ndim != 2 or knt_np.shape[1] != 3:
        raise ValueError("knt must be (N,3).")
    if ijk_np.ndim != 2:
        raise ValueError("ijk must be 2D (E,4) or (E,5).")
    if ijk_np.shape[1] == 4:
        # Append default mat_id = 1
        mat = np.ones((ijk_np.shape[0], 1), dtype=np.int32)
        ijk_np = np.hstack([ijk_np.astype(np.int64, copy=False), mat])
    elif ijk_np.shape[1] != 5:
        raise ValueError("ijk must have 4 or 5 columns (n1,n2,n3,n4[,mat_id]).")

    # Auto-detect one-based indexing only if not provided
    if detect_one_based is None:
        # One-based is a strong guess if min==1 and max==N (0-based would be max<=N-1).
        conn = ijk_np[:, :4]
        mn = int(conn.min())
        mx = int(conn.max())
        N = knt_np.shape[0]
        one_based = (mn == 1) and (mx == N)
    else:
        one_based = bool(detect_one_based)

    # Convert to JAX arrays (no in-place changes to user's arrays)
    knt = jnp.asarray(knt_np, dtype=jnp.float64)
    ijk = jnp.asarray(ijk_np, dtype=jnp.int32)

    # Precompute geometry
    geom = precompute_linear_tet_geometry(
        knt,
        ijk,
        one_based_indexing=one_based,
        fix_orientation=bool(fix_orientation),
    )
    return knt, geom


# ------------------------------- Updated loader: now delegates to the in-memory precompute function
def load_and_precompute_geometry(mesh_path: str, *, mmap: bool = True):
    """
    Load a tetrahedral mesh (.npz) and return precomputed TetGeom.

    Parameters
    ----------
    mesh_path : str
        Path to .npz file containing:
        - 'knt': (N,3) node coordinates
        - 'ijk': (E,4) or (E,5) connectivity [n1 n2 n3 n4 (opt: mat_group)]
    mmap : bool
        If True (default), use memory-mapping to reduce peak RAM during load.

    Returns
    -------
    knt : jnp.ndarray (N,3)
    geom : TetGeom
        Precomputed geometry (conn, grad_phi, volume, mat_id, volume_scalefactor).
    """
    data = np.load(mesh_path, mmap_mode="r" if mmap else None)
    try:
        knt_np = data["knt"]
        ijk_np = data["ijk"]
    finally:
        # np.load with mmap keeps the file open until the object is GC’d; leaving it
        # in scope is fine, but we don't hold references after conversion/precompute.
        pass

    # Delegate to the new in-memory precompute
    knt, geom = precompute_geometry_from_knt_ijk(
        knt_np,
        ijk_np,
        detect_one_based=None,  # auto-detect safely
        fix_orientation=True,
    )
    return knt, geom
