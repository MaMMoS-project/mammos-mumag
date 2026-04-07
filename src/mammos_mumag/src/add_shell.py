#!/usr/bin/env python3
from __future__ import annotations
"""
Add graded tetrahedral layers *outside* an existing body mesh using MeshPy (TetGen),
with flexible control of outermost geometric scale and automatic derivation of
either the per-layer geometric factor K or the number of layers L.

This module provides a pure in-memory API:
    run_add_shell_pipeline(...):  returns (knt, ijk) for the merged (body+shell) mesh
                                  and does **not** write any files.

Key features
------------
- Preserves the original body's surface triangles as the *inner* interface by
  supplying them as PLC facets and using TetGen region attributes and volume constraints.
- Builds homothetic shells S_l = K^l * S_0, l=0..L, and per-layer region seeds with
  per-region max volume constraints (derived from target h_l).
- Ties mesh size scaling to geometry with exponent beta: h_l = h0 * (K**beta)^(l+1),
  unless you give --hmax explicitly (then growth is inferred from hmax/h0).
- Merges body & shell, welds duplicate nodes (0-based, consecutive IDs), reorients tets
  to positive volume, and removes degenerate/duplicate elements.

Inputs (NPZ)
------------
knt : (N,3) float64
ijk : (E,4) or (E,5) int (optional mat_id in column 4)

Outputs (in-memory)
-------------------
knt : fused nodes (float64)
ijk : valid tets with mat_id (int32)

Requirements
------------
- meshpy (TetGen interface) -> pip install meshpy
- TetGen manual on PLC, quality, attributes, volume constraints:
  https://www.wias-berlin.de/software/tetgen/1.5/doc/manual
"""
import argparse
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
from meshpy.tet import MeshInfo, Options, build  # MeshPy -> TetGen

# Optional: VTU export
try:
    import meshio
    HAVE_meshio = True
except Exception:
    HAVE_meshio = False


# ------------------------------- utilities -------------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def parse_csv3(s: str) -> Tuple[float, float, float]:
    a = [float(x) for x in s.split(",")]
    if len(a) != 3:
        raise ValueError("Expected 'x,y,z'.")
    return float(a[0]), float(a[1]), float(a[2])


def approx_max_volume_from_edge(h: float) -> float:
    # Heuristic upper bound usable for TetGen -a (max volume per region)
    return 0.1 * (h ** 3)


def find_outer_surface_tris(ijk: np.ndarray) -> np.ndarray:
    """Return outer boundary faces (T,3) of the whole mesh from tet connectivity."""
    t4 = ijk[:, :4].astype(np.int64)
    fp = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    faces = t4[:, fp].reshape(-1, 3)
    keys = np.sort(faces, axis=1)
    uniq, inv, cnt = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return faces[cnt[inv] == 1]


def weld_points(
    knt: np.ndarray,
    ijk: np.ndarray,
    tol: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fuse nodes closer than 'tol' using integer grid hashing.
    Returns (knt_weld, ijk_remap, n_merged). Node IDs become 0..N-1 (no gaps).
    """
    if knt.size == 0:
        return knt, ijk, 0

    key = np.round(knt / tol).astype(np.int64)
    uniq_key, first_idx = np.unique(key, axis=0, return_index=True)
    id_of_key = {tuple(uniq_key[i]): i for i in range(uniq_key.shape[0])}
    map_old_to_new = np.array([id_of_key[tuple(k)] for k in key], dtype=np.int64)

    knt_weld = knt[first_idx]
    ijk4 = map_old_to_new[ijk[:, :4].astype(np.int64)]

    if ijk.shape[1] >= 5:
        ijk_out = np.hstack([ijk4.astype(np.int32), ijk[:, 4:5].astype(np.int32)])
    else:
        ijk_out = ijk4.astype(np.int32)

    n_merged = knt.shape[0] - knt_weld.shape[0]
    return knt_weld, ijk_out, n_merged


def orient_tets_positive(knt: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Swap two vertices where needed to ensure positive volume."""
    if tets.size == 0:
        return tets
    t = tets.copy()
    a, b, c, d = knt[t[:, 0]], knt[t[:, 1]], knt[t[:, 2]], knt[t[:, 3]]
    vols6 = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)
    bad = vols6 < 0.0
    if np.any(bad):
        tmp = t[bad].copy()
        tmp[:, [1, 2]] = tmp[:, [2, 1]]
        t[bad] = tmp
    return t


def remove_degenerate_and_duplicate_tets(
    knt: np.ndarray,
    ijk: np.ndarray,
    vol_eps: float = 1e-20
) -> np.ndarray:
    """Drop tets with repeated nodes or near-zero volume; drop duplicates (same node set)."""
    t4 = ijk[:, :4].astype(np.int64)
    unique_nodes = np.array([len(set(row)) == 4 for row in t4], dtype=bool)

    a, b, c, d = knt[t4[:, 0]], knt[t4[:, 1]], knt[t4[:, 2]], knt[t4[:, 3]]
    vols6 = np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a))
    keep = unique_nodes & (vols6 > vol_eps)
    ijk_kept = ijk[keep]
    if ijk_kept.shape[0] == 0:
        return ijk_kept

    t4s = np.sort(ijk_kept[:, :4], axis=1)
    uniq, first_idx = np.unique(t4s, axis=0, return_index=True)
    return ijk_kept[np.sort(first_idx)]


# -------------------------- surface size estimate --------------------------

def estimate_body_h_from_surface(knt: np.ndarray, ijk_with_mat: np.ndarray) -> float:
    """
    Estimate characteristic surface mesh size as the median boundary-edge length.
    """
    tris = find_outer_surface_tris(ijk_with_mat)
    if tris.size == 0:
        diag = np.linalg.norm(knt.max(0) - knt.min(0))
        return max(1e-6, 0.02 * diag)

    edges = np.vstack([
        np.sort(tris[:, [0, 1]], axis=1),
        np.sort(tris[:, [1, 2]], axis=1),
        np.sort(tris[:, [2, 0]], axis=1),
    ])
    edges = np.unique(edges, axis=0)
    p = knt[edges[:, 0]]
    q = knt[edges[:, 1]]
    lens = np.linalg.norm(q - p, axis=1)
    return max(float(np.median(lens)), 1e-12)


# ------------------------------- PLC builders -------------------------------

def build_layer_nodes(
    knt0: np.ndarray,
    surf_verts: np.ndarray,
    center: np.ndarray,
    K: float,
    layers: int
) -> Tuple[np.ndarray, Dict[Tuple[int, int], int], List[np.ndarray]]:
    """
    Create node copies for each surface vertex across layers l=0..L at scales K^l.
    node_map[(vid, l)] -> node id (with l=0 reusing original vertex id).
    """
    knt = knt0.copy()
    
    vmin = np.min(knt0, axis=0)
    vmax = np.max(knt0, axis=0)
    ext = vmax - vmin
    Lmax = float(np.max(ext))
    
    node_map: Dict[Tuple[int, int], int] = {}

    for vid in surf_verts:
        node_map[(int(vid), 0)] = int(vid)

    c = center.reshape(1, 3)
    v0 = (knt0[surf_verts] - c)  # rays from center to surface verts

    max_anisotropy = 5.
    
    # Print header once (before your loop)
    print(f"{'Layer':>5} | {'sx':>10} | {'sy':>10} | {'sz':>10}")
    print("-"*5 + "-+-" + "-"*10 + "-+-" + "-"*10 + "-+-" + "-"*10)

    svecs_layer = []
    for l in range(1, layers + 1):
        s = (K ** l)
        t = l / layers if layers > 1 else 1.0
        L_target = s * Lmax
        sx = np.array([L_target / ext[0], L_target / ext[1], L_target / ext[2]], dtype=float)
        sv = (1-t)*np.array([1.0,1.0,1.0]) + t*sx

        r = sv.max() / max(sv.min(), 1e-12)
        if r > max_anisotropy:
            # softly clamp: scale towards the geometric mean
            g = np.exp(np.log(sv).mean())
            # blend towards isotropy keeping product constant
            alpha = (r / max_anisotropy)  # >1
            sv = g * (sv / g) ** (1.0 / alpha)

        #pts = c + s * v0
        print(f"{l:5d} | {sv[0]:10.6f} | {sv[1]:10.6f} | {sv[2]:10.6f}")
        pts = c + v0 * sv.reshape(1, 3)
        start = knt.shape[0]
        knt = np.vstack([knt, pts])
        for i, vid in enumerate(surf_verts):
            node_map[(int(vid), l)] = start + i
            
        svecs_layer.append(sv.copy()) 

    return knt, node_map, svecs_layer


def make_shell_plc_from_surface(
    knt0: np.ndarray,
    tris0: np.ndarray,
    layers: int,
    K: float,
    center: Tuple[float, float, float]
):
    """
    Build a MeshPy/TetGen PLC with nested homothetic surfaces:
    S_0 (original), S_l = K^l * S_0 (l=1..L).
    All surfaces are added as triangular PLC facets. The interior of S_0 is marked as a 'hole'.
    """
    center = np.asarray(center, dtype=np.float64)
    tris0 = np.sort(tris0.astype(np.int64), axis=1)
    surf_verts = np.unique(tris0.reshape(-1))

    knt_all, node_map, svecs_layer = build_layer_nodes(knt0, surf_verts, center, K, layers)

    facets: List[List[int]] = []
    for l in range(0, layers + 1):
        for tri in tris0:
            v = [
                node_map[(int(tri[0]), l)],
                node_map[(int(tri[1]), l)],
                node_map[(int(tri[2]), l)],
            ]
            facets.append(v)

    # in make_shell_plc_from_surface(...), after knt_all/node_map are ready:
    tri0 = tris0[0]  # pick a stable triangle
    def centroid_at(layer: int) -> np.ndarray:
        vids = [node_map[(int(v), layer)] for v in tri0]
        return knt_all[vids].mean(axis=0)

    seeds = np.vstack([
        0.5 * (centroid_at(l) + centroid_at(l + 1))     # one seed per layer
        for l in range(layers)
    ]).astype(np.float64)

    # Optional sanity check:
    assert seeds.shape == (layers, 3)


    return knt_all, facets, seeds, surf_verts, node_map, svecs_layer


# ------------------------------ meshing core ------------------------------

def add_shell_with_meshpy(
    knt0: np.ndarray,
    ijk0: np.ndarray,
    layers: int,
    K: float,
    beta: float,
    center: Tuple[float, float, float],
    h0: float | None,
    hmax: float | None,
    minratio: float,
    max_steiner: int | None,
    no_exact: bool,
    verbose: bool,
    same_scaling: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create PLC shells and call TetGen (via MeshPy) to mesh only the exterior shell.
    Then merge with the original body mesh and validate/clean.
    """
    # Ensure we have mat_id column
    if ijk0.shape[1] >= 5:
        body_mat = int(ijk0[:, 4].max())
        ijk_body = ijk0[:, :5].astype(np.int32)
    else:
        body_mat = 1
        ijk_body = np.hstack([ijk0[:, :4].astype(np.int32),
                              np.full((ijk0.shape[0], 1), body_mat, dtype=np.int32)])
    shell_mat = body_mat + 1

    tris0 = find_outer_surface_tris(ijk_body)
    if tris0.size == 0:
        raise RuntimeError("Could not find outer surface triangles.")

    # PLC creation
    knt_plc, facets, seeds, surf_verts, node_map, svecs_layer = make_shell_plc_from_surface(
        knt0, tris0, layers=layers, K=K, center=center
    )

    mi = MeshInfo()
    mi.set_points(knt_plc.tolist())
    mi.set_facets(facets)  # PLC facets (triangles)

    # Mesh only the *interior of shells* outside body: mark body interior as a hole
    body_centroid = np.mean(knt0[surf_verts], axis=0)
    mi.set_holes([tuple(body_centroid)])

    # ---- Per-layer size schedule ----
    if same_scaling:
        beta = 1.0

    if h0 is None:
        raise RuntimeError("Internal error: h0 must be resolved before calling add_shell_with_meshpy.")


    # New: drive h_l from the actual sv used for S_{l+1}
    gms = [float(np.exp(np.log(sv).mean())) for sv in svecs_layer]  # geometric means (length 'layers')

    if hmax is None:
        scale = 1.0
    else:
        # ensure the *outermost* region hits hmax
        denom = max(h0 * (gms[-1] ** beta), 1e-30)
        scale = float(hmax) / denom

    mi.regions.resize(layers)
    for l in range(layers):
        h_l = scale * h0 * (gms[l] ** beta)      # region l is between S_l and S_{l+1}
        max_vol = approx_max_volume_from_edge(h_l)
        x, y, z = seeds[l]
        mi.regions[l] = (float(x), float(y), float(z), float(shell_mat), float(max_vol))

    # TetGen options: PLC (-p), quality (-q), region attrs (-A), volume (-a)
    switches = "pqAaY"  # -Y preserve PLC facets
    opts = Options(switches)
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)
    if max_steiner is not None:
        opts.parse_switches(f"S{int(max_steiner)}")
    if no_exact:
        opts.parse_switches("X")

    shell = build(mi, options=opts, attributes=True, volume_constraints=True, verbose=bool(verbose))
    knt_shell = np.asarray(shell.points, dtype=np.float64)
    tets_shell = np.asarray(shell.elements, dtype=np.int64)

    # Region attributes may be absent; in either case, set material id to shell_mat
    mat_shell = np.full(tets_shell.shape[0], shell_mat, dtype=np.int32)
    ijk_shell = np.hstack([tets_shell.astype(np.int32), mat_shell.reshape(-1, 1)])

    # Merge body + shell: append then weld
    knt_all = np.vstack([knt0, knt_shell])
    ijk_shell_shifted = ijk_shell.copy()
    ijk_shell_shifted[:, :4] += knt0.shape[0]
    ijk_merged = np.vstack([ijk_body, ijk_shell_shifted])

    knt_weld, ijk_weld, n_fused = weld_points(knt_all, ijk_merged, tol=1e-12)
    if n_fused:
        log(f"[validate] welded duplicate nodes: {n_fused}")

    ijk_weld[:, :4] = orient_tets_positive(knt_weld, ijk_weld[:, :4])
    ijk_weld = remove_degenerate_and_duplicate_tets(knt_weld, ijk_weld)
    return knt_weld, ijk_weld


# ----------------------------------------------------------------------
# Public entry point (importable) – pure in-memory (no writes)
# ----------------------------------------------------------------------
def run_add_shell_pipeline(
    *,
    in_npz: str,
    # Geometry controls
    layers: Optional[int] = None,
    K: Optional[float] = None,
    KL: Optional[float] = None,
    auto_layers: bool = False,
    auto_K: bool = False,
    # Mesh-size coupling
    beta: float = 1.0,
    same_scaling: bool = False,
    # Radial center
    center: Union[str, Tuple[float, float, float]] = "0,0,0",
    # Size targets
    h0: Optional[float] = None,
    hmax: Optional[float] = None,
    body_h: Optional[float] = None,
    # TetGen options
    minratio: float = 1.4,
    max_steiner: Optional[int] = None,
    no_exact: bool = False,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Programmatic equivalent of the CLI: load input NPZ, compute geometry/size
    parameters, build and merge the shell mesh, and return arrays in memory.

    Returns
    -------
    knt, ijk : np.ndarray
        Fused nodes and tetrahedra (ijk has mat_id in column 4).
    """
    # Load input
    data = np.load(in_npz)
    if "knt" not in data or "ijk" not in data:
        raise KeyError("Input NPZ must contain 'knt' and 'ijk'.")
    knt0 = data["knt"].astype(np.float64)
    ijk0 = data["ijk"].astype(np.int64)

    # Center parsing (supports str "cx,cy,cz" or tuple)
    if isinstance(center, str):
        cx, cy, cz = parse_csv3(center)
    else:
        cx, cy, cz = (float(center[0]), float(center[1]), float(center[2]))

    # Ensure we have a mat_id column for surface processing where needed
    if ijk0.shape[1] >= 5:
        ijk_body = ijk0[:, :5].astype(np.int32)
    else:
        ijk_body = np.hstack(
            [ijk0[:, :4].astype(np.int32), np.ones((ijk0.shape[0], 1), dtype=np.int32)]
        )  # dummy mat

    # ---- Determine geometry triplet (L, K, KL) based on flags ----
    if auto_layers and auto_K:
        raise ValueError("Choose only one: auto_layers or auto_K.")

    L = layers
    K_val = K
    KL_val = KL

    if auto_layers:
        if (KL_val is None) or (K_val is None):
            raise ValueError("auto_layers requires both KL and K.")
        if KL_val <= 1.0 or K_val <= 1.0:
            raise ValueError("Require KL>1 and K>1 for auto_layers.")
        L = max(1, int(round(math.log(KL_val) / math.log(K_val))))
        if verbose:
            log(f"[geom] auto-layers: KL={KL_val:g}, K={K_val:g} -> layers L={L}")
    elif auto_K:
        if (KL_val is None) or (L is None):
            raise ValueError("auto_K requires both KL and layers.")
        if KL_val <= 1.0 or L < 1:
            raise ValueError("Require KL>1 and layers>=1.")
        K_val = KL_val ** (1.0 / L)
        if verbose:
            log(f"[geom] auto-K: KL={KL_val:g}, L={L} -> K={K_val:.8g}")
    else:
        # Neither auto; require at least L and K
        if (L is None) or (K_val is None):
            raise ValueError(
                "Provide layers and K, or use auto_layers (KL & K) or auto_K (KL & layers)."
            )
        if L < 1 or K_val <= 1.0:
            raise ValueError("Require layers>=1 and K>1.")

    # ---- Mesh-size defaults from body surface ----
    if body_h is None:
        body_h_val = estimate_body_h_from_surface(knt0, ijk_body)
        if verbose:
            log(f"[size] Derived body_h (median boundary-edge): {body_h_val:.6g}")
    else:
        body_h_val = float(body_h)
        if verbose:
            log(f"[size] Using user-provided body_h: {body_h_val:.6g}")

    if h0 is None:
        h0_val = 1.5 * body_h_val
        if verbose:
            log(f"[size] h0 not given -> set h0 = 1.5 * body_h = {h0_val:.6g}")
    else:
        h0_val = float(h0)
        if verbose:
            log(f"[size] Using user-provided h0: {h0_val:.6g}")

    # If hmax not provided and same_scaling is requested, set hmax = h0 * (K^L)
    if (hmax is None) and same_scaling:
        hmax_val = h0_val * (K_val ** (L))
        if verbose:
            log(f"[size] same-scaling: set hmax = h0 * K^L = {hmax_val:.6g}")
    else:
        hmax_val = (float(hmax) if (hmax is not None) else None)

    if verbose:
        log(
            f"[info] Geometry: L={L}, K={K_val:.6g}, "
            f"KL(req)={'-' if KL_val is None else f'{KL_val:.6g}'}, KL(achieved)={(K_val**L):.6g}"
        )
    if verbose:
        log(
            f"[info] Mesh-size: beta={1.0 if same_scaling else beta}, "
            f"h0={h0_val:.6g}, hmax={'(derived)' if hmax_val is None else f'{hmax_val:.6g}'}"
        )
        log(
            f"[info] center=({cx},{cy},{cz}), minratio={minratio}, "
            f"max_steiner={'-' if max_steiner is None else max_steiner}, no_exact={bool(no_exact)}"
        )

    # ---- Build shells & mesh them ----
    knt, ijk = add_shell_with_meshpy(
        knt0,
        ijk0,
        layers=L,
        K=K_val,
        beta=(1.0 if same_scaling else beta),
        center=(cx, cy, cz),
        h0=h0_val,
        hmax=hmax_val,
        minratio=minratio,
        max_steiner=max_steiner,
        no_exact=bool(no_exact),
        verbose=bool(verbose),
        same_scaling=bool(same_scaling),
    )
    print("layers = ",L," nodes = ",len(knt)," elements = ",len(ijk))
    return knt, ijk


# ------------------------------ CLI (optional) ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Add graded exterior tetrahedral layers using MeshPy/TetGen (in-memory).")
    ap.add_argument("--in", dest="in_npz", required=True, help="Input NPZ with knt, ijk")
    ap.add_argument("--layers", type=int, default=None, help="Number of layers L (>=1)")
    ap.add_argument("--K", type=float, default=None, help="Per-layer geometric scale (>1)")
    ap.add_argument("--KL", type=float, default=None, help="Outermost geometric scale S_L = KL*S_0 (>1)")
    ap.add_argument("--auto-layers", action="store_true",
                    help="Given KL and K, compute L = round(log(KL)/log(K)). Requires --KL and --K.")
    ap.add_argument("--auto-K", action="store_true",
                    help="Given KL and L, compute K = KL**(1/L). Requires --KL and --layers.")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="Mesh-size/geometry coupling exponent (default 1.0 -> same scaling)")
    ap.add_argument("--same-scaling", action="store_true",
                    help="Shortcut: enforce beta=1 and (if --hmax missing) set hmax=h0*K**(layers)")
    ap.add_argument("--center", type=str, default="0,0,0", help="Ray origin 'cx,cy,cz' (default 0,0,0)")
    ap.add_argument("--h0", type=float, default=None, help="Target edge near body (first shell). Default: 1.5*body_h")
    ap.add_argument("--hmax", type=float, default=None, help="Target edge at outermost shell. Default: derived")
    ap.add_argument("--body-h", type=float, default=None,
                    help="If omitted, derived as median boundary-edge length on the body surface.")
    ap.add_argument("--minratio", type=float, default=1.4, help="Tet quality minratio for TetGen -q")
    ap.add_argument("--max-steiner", type=int, default=None, help="Limit Steiner points (TetGen -S#)")
    ap.add_argument("--no-exact", action="store_true", help="TetGen -X (suppress exact arithmetic)")
    ap.add_argument("--verbose", action="store_true")

    # NEW: optional VTU export of the merged (body + shells) mesh
    ap.add_argument("--out-npz", type=str, default=None,
                    help="Optional path to write merged mesh as npz ('.vtu' auto-added).")
    ap.add_argument("--out-vtu", type=str, default=None,
                    help="Optional path to write merged mesh as VTU ('.vtu' auto-added). Requires meshio.")

    args = ap.parse_args()

    knt, ijk = run_add_shell_pipeline(
        in_npz=args.in_npz,
        layers=args.layers,
        K=args.K,
        KL=args.KL,
        auto_layers=bool(args.auto_layers),
        auto_K=bool(args.auto_K),
        beta=float(args.beta),
        same_scaling=bool(args.same_scaling),
        center=args.center,  # string "cx,cy,cz" is accepted
        h0=args.h0,
        hmax=args.hmax,
        body_h=args.body_h,
        minratio=float(args.minratio),
        max_steiner=args.max_steiner,
        no_exact=bool(args.no_exact),
        verbose=bool(args.verbose),
    )
    print(f"[ok] merged in-memory mesh: nodes={knt.shape[0]:,}, tets={ijk.shape[0]:,}")

    # Optional VTU export
    if args.out_vtu:
        out_vtu = str(Path(args.out_vtu).with_suffix(".vtu"))
        if not HAVE_meshio:
            print("[warn] meshio not installed; skipping VTU export. Install with: pip install meshio")
        else:
            cells = [("tetra", ijk[:, :4].astype(np.int32))]
            cell_data = {"mat_id": [ijk[:, 4].astype(np.int32)]}
            meshio.Mesh(points=knt.astype(np.float64), cells=cells, cell_data=cell_data).write(out_vtu)
            print(f"[ok] wrote VTU -> {out_vtu}")

    if args.out_npz:
        out_npz = str(Path(args.out_npz).with_suffix(".npz"))
        np.savez(out_npz, knt=knt.astype(np.float64), ijk=ijk.astype(np.int32))


if __name__ == "__main__":
    main()
