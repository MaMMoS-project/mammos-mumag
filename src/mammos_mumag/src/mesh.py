from __future__ import annotations
import sys
import argparse
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import numpy as np
from scipy.spatial import Delaunay
#!/usr/bin/env python3
"""
Single-solid tetra mesher with selectable geometry and backend:
- Geometry: box (parallelepiped) or ellipsoid (symmetry axis is local z).
- Backends:
  * meshpy (TetGen): quality/volume constrained tetrahedralization.
  * grid: regular brick grid, each brick split into 6 tets (Freudenthal).

New in this version:
- --ell-subdiv auto/automatic/-1 (or even the misspelling 'uatomatic'):
  Automatically selects an icosphere subdivision level based on h and size.
- Ellipsoid orientation using --dir-x, --dir-y, --dir-z:
  The ellipsoid symmetry axis is the local z-axis; these flags orient the ellipsoid
  in 3D space. Box already supported orientation; now ellipsoid does too.

Features:
- Extents Lx,Ly,Lz and mesh size h.
- Box: optional orientation via dir-x, dir-y, dir-z.
- Ellipsoid: axisymmetric with a=b=(Lx+Ly)/2 in the local xy-plane, c=Lz/2 along local z,
  then oriented using the provided frame.
- Centered at origin by construction.
- Saves .npz (knt, ijk) and .vtu (visualization), mat_id=1 for all tets.

Dependencies:
- For meshpy backend: meshpy (TetGen) -> pip install meshpy
- For .vtu export: meshio -> pip install meshio
- Grid backend works without meshpy; visualization still needs meshio.
"""




# Optional: visualization
try:
    import meshio

    HAVE_meshio = True
except Exception:
    HAVE_meshio = False

# Optional: TetGen backend
try:
    from meshpy.tet import MeshInfo, Options, build as tet_build

    HAVE_meshpy = True
except Exception:
    HAVE_meshpy = False

# ------------------------------- Utilities -------------------------------


def parse_csv3(s: str) -> Tuple[float, float, float]:
    vals = [float(x) for x in s.split(",")]
    if len(vals) != 3:
        raise ValueError("Expected three comma-separated values, e.g. '1,0,0'.")
    return float(vals[0]), float(vals[1]), float(vals[2])


def with_ext(path_like: str, ext: str) -> str:
    p = Path((path_like or "").strip() or "single_solid")
    if p.suffix.lower() != ext.lower():
        p = p.with_suffix(ext)
    return str(p)


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("Zero-length direction vector is not allowed.")
    return v / n


def orthonormal_frame(
    xdir: Tuple[float, float, float],
    ydir: Tuple[float, float, float],
    zdir: Tuple[float, float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Gram–Schmidt orthonormalization, enforce right-handedness.
    """
    x = normalize(np.asarray(xdir, dtype=float))
    y = np.asarray(ydir, dtype=float)
    y = y - np.dot(y, x) * x
    if np.linalg.norm(y) < 1e-12:
        raise ValueError("dir-y is colinear with dir-x; provide independent vectors.")
    y = normalize(y)
    z = np.asarray(zdir, dtype=float)
    z = z - np.dot(z, x) * x - np.dot(z, y) * y
    if np.linalg.norm(z) < 1e-12:
        z = np.cross(x, y)
    z = normalize(z)
    R = np.column_stack((x, y, z))
    if np.linalg.det(R) < 0:
        z = -z
    return x, y, z


def approx_max_volume_from_edge(h: float) -> float:
    # Practical heuristic for TetGen's max volume from target edge length ~h
    return 0.1 * (h**3)


# ------------------------------- Geometry: BOX -------------------------------


def oriented_point(
    x: float, y: float, z: float, ex: np.ndarray, ey: np.ndarray, ez: np.ndarray
) -> np.ndarray:
    return x * ex + y * ey + z * ez


def oriented_box_facets(
    points: List[Tuple[float, float, float]],
    center: Tuple[float, float, float],
    half: Tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
) -> List[List[int]]:
    cx, cy, cz = center
    hx, hy, hz = half
    c = np.array([cx, cy, cz], dtype=float)
    signs = [
        (-1, -1, -1),
        (+1, -1, -1),
        (+1, +1, -1),
        (-1, +1, -1),
        (-1, -1, +1),
        (+1, -1, +1),
        (+1, +1, +1),
        (-1, +1, +1),
    ]
    base = len(points)
    for sx, sy, sz in signs:
        v = c + (sx * hx) * ex + (sy * hy) * ey + (sz * hz) * ez
        points.append(tuple(v.tolist()))
    faces = [
        [base + 0, base + 1, base + 2, base + 3],  # bottom
        [base + 4, base + 5, base + 6, base + 7],  # top
        [base + 1, base + 5, base + 6, base + 2],  # +x
        [base + 0, base + 3, base + 7, base + 4],  # -x
        [base + 3, base + 2, base + 6, base + 7],  # +y
        [base + 0, base + 4, base + 5, base + 1],  # -y
    ]
    return faces


# ------------------------------- Geometry: ELLIPSOID -------------------------------


def icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    """Return (V,F) for a unit icosahedron centered at origin."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        ],
        dtype=float,
    )
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    return verts, faces


def subdivide_icosphere(
    verts: np.ndarray, faces: np.ndarray, level: int = None, subdiv: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loop-subdivide each triangle into 4, projecting new verts back to unit sphere.
    Accepts either 'level' (positional/keyword) or 'subdiv' (keyword) for convenience.
    """
    # Normalize input level
    if subdiv is None and level is None:
        lvl = 0
    elif subdiv is None:
        lvl = int(level)
    else:
        lvl = int(subdiv)
    lvl = max(lvl, 0)

    def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        m = (a + b) * 0.5
        return m / np.linalg.norm(m)

    V = np.asarray(verts, dtype=float)
    F = np.asarray(faces, dtype=np.int32)

    for _ in range(lvl):
        edge_cache: Dict[Tuple[int, int], int] = {}
        new_faces = []
        new_verts = V.tolist()

        def mid_idx(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in edge_cache:
                return edge_cache[key]
            vi, vj = np.array(new_verts[i]), np.array(new_verts[j])
            vm = midpoint(vi, vj)
            new_verts.append(vm.tolist())
            idx = len(new_verts) - 1
            edge_cache[key] = idx
            return idx

        for i, j, k in F:
            a = mid_idx(i, j)
            b = mid_idx(j, k)
            c = mid_idx(k, i)
            new_faces.extend(
                [
                    (i, a, c),
                    (a, j, b),
                    (c, b, k),
                    (a, b, c),
                ]
            )
        V = np.asarray(new_verts, dtype=float)
        F = np.asarray(new_faces, dtype=np.int32)
        # Ensure on unit sphere
        norms = np.linalg.norm(V, axis=1)
        V = V / norms[:, None]
    return V, F


def ellipsoid_surface(
    extents: Tuple[float, float, float], subdiv: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a near-uniform triangular surface of an ellipsoid centered at origin in LOCAL coords.
    Rotational symmetry in the local xy-plane is enforced: a=b=(Lx+Ly)/2, c=Lz/2.
    Returns local vertices V (Nv,3) and faces F (Nf,3).
    """
    import sys as _sys

    Lx, Ly, Lz = extents
    if not (Lx > 0 and Ly > 0 and Lz > 0):
        raise ValueError("All ellipsoid extents must be positive.")
    if abs(Lx - Ly) > 1e-12:
        print(
            f"[warn] Enforcing rotational symmetry: Lx({Lx}) != Ly({Ly}). Using average in xy.",
            file=_sys.stderr,
        )
    Lxy = 0.5 * (Lx + Ly)
    a = Lxy / 2.0
    b = Lxy / 2.0
    c = Lz / 2.0

    V0, F0 = icosahedron()
    # Works with either version of subdivide_icosphere (level or subdiv keyword)
    V, F = subdivide_icosphere(V0, F0, subdiv=max(int(subdiv), 0))

    # Scale to ellipsoid in LOCAL coordinates
    V = np.ascontiguousarray(
        np.column_stack((a * V[:, 0], b * V[:, 1], c * V[:, 2])), dtype=np.float64
    )
    F = np.asarray(F, dtype=np.int32)
    return V, F


# ------------------------------- Auto ellipsoid subdivision -------------------------------


def auto_ell_subdiv(
    Lx: float, Ly: float, Lz: float, h: float, kappa: float = 1.0
) -> int:
    """
    Choose a surface subdivision level so average surface edge length ~ kappa * h.
    Use R = max(a, c) with a=b=(Lx+Ly)/2 and c=Lz/2.
    Empirical average edge length on a unit sphere after n subdivisions ~ 1.20 / 2^n.
    """
    if not (Lx > 0 and Ly > 0 and Lz > 0 and h > 0):
        return 0
    a = 0.5 * (0.5 * (Lx + Ly))  # radius a=b=Lxy/2 -> a = 0.25*(Lx+Ly)
    c = 0.5 * Lz
    R = max(a, c)
    n = int(np.ceil(np.log2((1.20 * R) / max(kappa * h, 1e-12))))
    return max(0, n)


def parse_ell_subdiv_option(
    val: str, Lx: float, Ly: float, Lz: float, h: float, kappa: float = 1.0
) -> int:
    """
    Accepts integers >=0, or 'auto'/'automatic'/'uatomatic', or '-1' for auto.
    """
    s = str(val).strip().lower()
    if s in ("auto", "automatic", "uatomatic", "-1"):
        return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)
    try:
        n = int(s)
        if n < 0:
            return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)
        return n
    except Exception:
        # Fallback to auto if parsing fails
        return auto_ell_subdiv(Lx, Ly, Lz, h, kappa=kappa)


# ------------------------------- Eye geometry (from generate_element.py) -------------------------------
def bezier_quad(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: float) -> np.ndarray:
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def sample_bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n)
    return np.array([bezier_quad(p0, p1, p2, t) for t in ts])


def build_eye_polygon(
    length: float = 3.5, width: float = 1.0, samples_per_curve: int = 64
) -> np.ndarray:
    """Return an Nx2 array with CCW polygon points of the eye shape in LOCAL XY coords.

    length: full length along x (Lx)
    width: half-height (peak y coordinate). If caller provides full Ly, pass width=Ly/2.
    """
    p_left = np.array([-length / 2.0, 0.0])
    p_top = np.array([0.0, width])
    p_right = np.array([length / 2.0, 0.0])

    top_curve = sample_bezier(p_left, p_top, p_right, samples_per_curve)
    bottom_curve = sample_bezier(
        p_right, np.array([0.0, -width]), p_left, samples_per_curve
    )

    polygon = np.vstack([top_curve, bottom_curve])
    return polygon


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Ray-casting point-in-polygon test. points: (M,2), polygon: (N,2). Returns boolean (M,).
    Uses the even-odd rule. Implemented in pure Python/numpy to avoid extra deps.
    """
    px = points[:, 0]
    py = points[:, 1]
    x = polygon[:, 0]
    y = polygon[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    n = polygon.shape[0]
    for i in range(n):
        j = (i + 1) % n
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        # edges where the horizontal ray intersects
        intersect = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-30) + xi
        )
        inside ^= intersect
    return inside


def triangulate_polygon(polygon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Delaunay triangulation of polygon points, return (points, triangles) with triangles filtered
    to lie inside the polygon using centroid test.
    """
    tri = Delaunay(polygon)
    triangles = tri.simplices
    centroids = polygon[triangles].mean(axis=1)
    mask = _points_in_polygon(centroids, polygon)
    triangles = triangles[mask]
    return polygon, triangles


# ------------------------------- Elliptic cylinder (extruded ellipse) -------------------------------


def build_ellipse_polygon(a: float = 1.0, b: float = 0.5, n: int = 128) -> np.ndarray:
    """Return an Nx2 CCW polygon approximating an axis-aligned ellipse with semi-axes a (x) and b (y)."""
    if a <= 0 or b <= 0:
        raise ValueError("Ellipse semi-axes must be positive.")
    ts = np.linspace(0.0, 2.0 * np.pi, max(8, int(n)), endpoint=False)
    x = a * np.cos(ts)
    y = b * np.sin(ts)
    polygon = np.column_stack((x, y))
    return polygon

def mesh_backend_meshpy_elliptic_cylinder(
    a: float,          # semi-axis along local x
    b: float,          # semi-axis along local y
    t: float,          # thickness along local z
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    """
    Mesh an elliptic cylinder (ellipse cross-section with semi-axes a, b; extruded by thickness t)
    using MeshPy/TetGen, without pre-triangulating the caps. The top and bottom faces are single
    N-gon facets; the side surface is N quads.
    """
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")

    # 1) Build boundary polygon (LOCAL XY) approximating the ellipse (CCW)
    polygon = build_ellipse_polygon(a=a, b=b, n=128)  # shape (N, 2)

    # 2) Build 3D vertices in LOCAL coords, then map to WORLD
    top_z, bottom_z = t / 2.0, -t / 2.0
    verts_top    = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])
    V_local = np.vstack([verts_top, verts_bottom])

    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # 3) Facets: top N-gon (CCW), bottom N-gon (reversed), side quads
    facets: list[list[int]] = []
    N = polygon.shape[0]

    # Top N-gon
    facets.append(list(range(0, N)))

    # Bottom N-gon (reverse)
    facets.append(list(range(2 * N - 1, N - 1, -1)))

    # Side quads
    for i in range(N):
        ni = (i + 1) % N
        aidx = i
        bidx = ni
        cidx = N + ni
        didx = N + i
        facets.append([aidx, bidx, cidx, didx])

    # 4) TetGen via MeshPy
    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)  # polygons & quads

    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )




'''
def mesh_backend_meshpy_elliptic_cylinder(
    a: float,
    b: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    # build polygon approximation and triangulate for top/bottom caps
    polygon = build_ellipse_polygon(a=a, b=b, n=128)
    polygon, triangles = triangulate_polygon(polygon)

    top_z = t / 2.0
    bottom_z = -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])

    V_local = np.vstack([verts_top, verts_bottom])
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    facets = []
    N = polygon.shape[0]
    for tri in triangles:
        facets.append([int(tri[0]), int(tri[1]), int(tri[2])])
    for tri in triangles:
        facets.append([int(N + tri[2]), int(N + tri[1]), int(N + tri[0])])
    for i in range(N):
        ni = (i + 1) % N
        aidx = i
        bidx = ni
        cidx = N + ni
        didx = N + i
        facets.append([aidx, bidx, cidx, didx])

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
'''

def mesh_backend_grid_elliptic_cylinder(
    a: float,
    b: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
):
    # bounding box: x in [-a,a], y in [-b,b], z in [-t/2,t/2]
    Lx = 2.0 * float(a)
    Ly = 2.0 * float(b)
    Lz = float(t)
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    Nnodes = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((Nnodes, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple] = []
    polygon = build_ellipse_polygon(a=a, b=b, n=128)

    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array(
            [np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64
        )

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                for tcell in local_tets:
                    P_world = knt[list(tcell), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    inside = _points_in_polygon(ctd[None, :2], polygon)[0]
                    if inside and abs(ctd[2]) <= (Lz / 2.0 + 1e-12):
                        tets.append(tcell)

    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        print(
            f"[info:grid:elliptic_cylinder] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}",
            flush=True,
        )
    return knt, ijk


# Mesh backends for eye geometry
def mesh_backend_meshpy_eye(
    length: float,
    width: float,     # NOTE: this is the half-height (i.e., Ly/2)
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    """
    Mesh the extruded 'eye' shape (built from two quadratic Bézier arcs) using MeshPy/TetGen,
    without pre-triangulating the caps. The top and bottom faces are passed as single N-gon facets;
    the side surface is passed as quads between successive boundary vertices.
    """
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")

    # 1) Build boundary polygon (LOCAL XY). build_eye_polygon returns CCW points
    polygon = build_eye_polygon(length=length, width=width)  # shape (N, 2)

    # 2) Build 3D vertices for top and bottom in LOCAL coords, then map to WORLD
    top_z, bottom_z = t / 2.0, -t / 2.0
    verts_top    = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])
    V_local = np.vstack([verts_top, verts_bottom])

    # Map LOCAL -> WORLD using orthonormal frame (ex, ey, ez)
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # 3) Build facets:
    #    - Top: one N-gon (0..N-1), keep CCW order for outward normal
    #    - Bottom: one N-gon (N..2N-1), use reversed order to maintain outward normal
    #    - Sides: N quads (a,b,c,d) wrapping around the ring
    facets: list[list[int]] = []
    N = polygon.shape[0]

    # Top N-gon
    facets.append(list(range(0, N)))

    # Bottom N-gon (reverse)
    facets.append(list(range(2 * N - 1, N - 1, -1)))

    # Side quads
    for i in range(N):
        ni = (i + 1) % N
        a = i
        b = ni
        c = N + ni
        d = N + i
        facets.append([a, b, c, d])

    # 4) TetGen via MeshPy
    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)  # polygons & quads; TetGen will triangulate them

    # Region with volume constraint derived from h
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    # TetGen options
    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )

    # Return nodes and tets with mat_id=1
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk

'''
def mesh_backend_meshpy_eye(
    length: float,
    width: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    polygon = build_eye_polygon(length=length, width=width)
    polygon, triangles = triangulate_polygon(polygon)

    # Build 3D points: top and bottom
    top_z = t / 2.0
    bottom_z = -t / 2.0
    verts_top = np.hstack([polygon, np.full((polygon.shape[0], 1), top_z)])
    verts_bottom = np.hstack([polygon, np.full((polygon.shape[0], 1), bottom_z)])

    # Map local to world using orientation frame
    V_local = np.vstack([verts_top, verts_bottom])
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    # facets: top triangles, bottom triangles (reversed), side quads
    facets = []
    N = polygon.shape[0]
    for tri in triangles:
        facets.append([int(tri[0]), int(tri[1]), int(tri[2])])
    for tri in triangles:
        # bottom triangles offset by N, reversed
        facets.append([int(N + tri[2]), int(N + tri[1]), int(N + tri[0])])
    # side quads (as 4-vertex facets)
    for i in range(N):
        ni = (i + 1) % N
        a = i
        b = ni
        c = N + ni
        d = N + i
        facets.append([a, b, c, d])

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
'''

def mesh_backend_grid_eye(
    length: float,
    width: float,
    t: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
):
    # Build local bounding box for the extruded eye: x in [-Lx/2,Lx/2], y in [-width,width], z in [-t/2,t/2]
    Lx = float(length)
    Ly = float(2.0 * width)
    Lz = float(t)
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    Nnodes = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((Nnodes, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: list[tuple] = []
    polygon = build_eye_polygon(length=length, width=width)

    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array(
            [np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64
        )

    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                for tcell in local_tets:
                    P_world = knt[list(tcell), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    # Test if centroid's (x,y) is inside 2D polygon and z within thickness
                    inside = _points_in_polygon(ctd[None, :2], polygon)[0]
                    if inside and abs(ctd[2]) <= (Lz / 2.0 + 1e-12):
                        tets.append(tcell)

    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        print(
            f"[info:grid:eye] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}",
            flush=True,
        )
    return knt, ijk


# ------------------------------- Backends -------------------------------


def mesh_backend_meshpy_box(
    extents: Tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    minratio: float,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    Lx, Ly, Lz = extents
    half = (0.5 * Lx, 0.5 * Ly, 0.5 * Lz)
    points: List[Tuple[float, float, float]] = []
    facets = oriented_box_facets(points, (0.0, 0.0, 0.0), half, ex, ey, ez)

    mi = MeshInfo()
    mi.set_points(points)
    mi.set_facets(facets)
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


def mesh_backend_meshpy_ellipsoid(
    extents: Tuple[float, float, float],
    h: float,
    minratio: float,
    subdiv: int,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    verbose: bool,
):
    if not HAVE_meshpy:
        raise RuntimeError("meshpy is not installed. Install with: pip install meshpy")
    # Build LOCAL ellipsoid surface then orient to world using (ex,ey,ez)
    V_local, F = ellipsoid_surface(extents, subdiv=subdiv)
    V_world = np.ascontiguousarray(
        V_local[:, 0:1] * ex[None, :]
        + V_local[:, 1:2] * ey[None, :]
        + V_local[:, 2:3] * ez[None, :],
        dtype=np.float64,
    )

    mi = MeshInfo()
    mi.set_points(V_world.tolist())
    mi.set_facets([list(tri) for tri in F.tolist()])  # triangles
    mi.regions.resize(1)
    mi.regions[0] = (0.0, 0.0, 0.0, 1.0, approx_max_volume_from_edge(float(h)))

    opts = Options("pqAa")
    opts.minratio = float(minratio)
    opts.regionattrib = True
    opts.verbose = bool(verbose)

    mesh = tet_build(
        mi,
        options=opts,
        attributes=True,
        volume_constraints=True,
        verbose=bool(verbose),
    )
    knt = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.elements, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk


def mesh_backend_grid_box(
    extents: Tuple[float, float, float],
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    h: float,
    verbose: bool,
):
    Lx, Ly, Lz = extents
    nx = max(1, int(np.ceil(Lx / h)))
    ny = max(1, int(np.ceil(Ly / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lx / 2, Lx / 2, nx + 1)
    ys = np.linspace(-Ly / 2, Ly / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    N = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((N, 3), dtype=np.float64)
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    tets: List[Tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                tets.extend(
                    [
                        (A, B, D, H),
                        (A, B, F, H),
                        (A, C, D, H),
                        (A, C, G, H),
                        (A, E, F, H),
                        (A, E, G, H),
                    ]
                )
    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        print(
            f"[info:grid:box] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, tets={ijk.shape[0]}",
            flush=True,
        )
    return knt, ijk


def mesh_backend_grid_ellipsoid(
    extents: Tuple[float, float, float],
    h: float,
    ex: np.ndarray,
    ey: np.ndarray,
    ez: np.ndarray,
    verbose: bool,
):
    """
    LOCAL brick grid fitted to the local bounding box; map nodes to world via (ex,ey,ez).
    Keep only tetrahedra whose centroid (projected back to local coords) is inside the
    local ellipsoid x^2/a^2 + y^2/b^2 + z^2/c^2 <= 1, with a=b=(Lx+Ly)/2, c=Lz/2.
    """
    import sys as _sys

    Lx, Ly, Lz = extents
    if abs(Lx - Ly) > 1e-12:
        print(
            f"[warn] Enforcing rotational symmetry: Lx({Lx}) != Ly({Ly}). Using average in xy.",
            file=_sys.stderr,
        )
    Lxy = 0.5 * (Lx + Ly)
    a, b, c = Lxy / 2.0, Lxy / 2.0, Lz / 2.0

    # Build LOCAL grid
    nx = max(1, int(np.ceil(Lxy / h)))
    ny = max(1, int(np.ceil(Lxy / h)))
    nz = max(1, int(np.ceil(Lz / h)))
    xs = np.linspace(-Lxy / 2, Lxy / 2, nx + 1)
    ys = np.linspace(-Lxy / 2, Lxy / 2, ny + 1)
    zs = np.linspace(-Lz / 2, Lz / 2, nz + 1)

    def nidx(i, j, k) -> int:
        return i + (nx + 1) * (j + (ny + 1) * k)

    N = (nx + 1) * (ny + 1) * (nz + 1)
    knt = np.empty((N, 3), dtype=np.float64)
    # map LOCAL nodes to WORLD coords
    for k in range(nz + 1):
        z = zs[k]
        for j in range(ny + 1):
            y = ys[j]
            for i in range(nx + 1):
                x = xs[i]
                p = oriented_point(x, y, z, ex, ey, ez)
                knt[nidx(i, j, k), :] = p

    # Helper to project WORLD point back to LOCAL coordinates (orthonormal frame)
    def to_local(pw: np.ndarray) -> np.ndarray:
        return np.array(
            [np.dot(pw, ex), np.dot(pw, ey), np.dot(pw, ez)], dtype=np.float64
        )

    tets: List[Tuple[int, int, int, int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                A = nidx(i, j, k)
                B = nidx(i + 1, j, k)
                C = nidx(i, j + 1, k)
                D = nidx(i + 1, j + 1, k)
                E = nidx(i, j, k + 1)
                F = nidx(i + 1, j, k + 1)
                G = nidx(i, j + 1, k + 1)
                H = nidx(i + 1, j + 1, k + 1)
                local_tets = [
                    (A, B, D, H),
                    (A, B, F, H),
                    (A, C, D, H),
                    (A, C, G, H),
                    (A, E, F, H),
                    (A, E, G, H),
                ]
                # keep only tets whose centroid is inside the LOCAL ellipsoid
                for t in local_tets:
                    P_world = knt[list(t), :]
                    ctd_world = P_world.mean(axis=0)
                    ctd = to_local(ctd_world)
                    val = (ctd[0] / a) ** 2 + (ctd[1] / b) ** 2 + (ctd[2] / c) ** 2
                    if val <= 1.0 + 1e-12:
                        tets.append(t)
    tets = np.asarray(tets, dtype=np.int32)
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    if verbose:
        print(
            f"[info:grid:ellipsoid] nx,ny,nz=({nx},{ny},{nz}); nodes={knt.shape[0]}, kept tets={ijk.shape[0]}",
            flush=True,
        )
    return knt, ijk


# ------------------------------- Programmatic entry point -------------------------------


def run_single_solid_mesher(
    *,
    geom: str = "box",  # "box" | "ellipsoid" | "eye"
    extent: Union[str, Tuple[float, float, float]] = "60.0,60.0,60.0",
    h: float = 2.0,
    minratio: float = 1.4,  # meshpy backend only
    backend: str = "meshpy",  # "meshpy" | "grid"
    dir_x: Union[str, Tuple[float, float, float]] = "1,0,0",
    dir_y: Union[str, Tuple[float, float, float]] = "0,1,0",
    dir_z: Union[str, Tuple[float, float, float]] = "0,0,1",  # ellipsoid symmetry axis
    ell_subdiv: Union[
        str, int
    ] = "auto",  # ellipsoid + meshpy: int >=0 or 'auto'/'automatic'/'-1'
    out_name: Optional[str] = "single_solid",
    out_data_name: Optional[str] = None,  # overrides .npz base name
    out_vis_name: Optional[str] = None,  # overrides .vtu base name
    number_of_grains=1,
    seed=123,
    no_vis: bool = False,
    verbose: bool = False,
    return_arrays: bool = True,  # NEW: set False to minimize memory
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Optional[str]]:
    """
    Build a single-solid tetrahedral mesh (box or ellipsoid) using either the meshpy (TetGen)
    or grid backend, write outputs (.npz and optional .vtu), and return:
        (knt, ijk, out_npz_path, out_vtu_path_or_none)

    Set return_arrays=False to avoid keeping (knt, ijk) in memory; only file paths are returned.

    Raises:
        ValueError / RuntimeError on invalid inputs or backend errors.
    """
    # Parse extents and orientation inputs
    if isinstance(extent, str):
        Lx, Ly, Lz = parse_csv3(extent)
    else:
        Lx, Ly, Lz = float(extent[0]), float(extent[1]), float(extent[2])

    def _csv_or_tuple(
        v: Union[str, Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        return (
            parse_csv3(v)
            if isinstance(v, str)
            else (float(v[0]), float(v[1]), float(v[2]))
        )

    dx = _csv_or_tuple(dir_x)
    dy = _csv_or_tuple(dir_y)
    dz = _csv_or_tuple(dir_z)

    # Build orthonormal frame from user directions (used for both shapes)
    ex, ey, ez = orthonormal_frame(dx, dy, dz)

    # TODO: include the geometry for a elliptic_cylinder shape as option next to box, eye and ellipsoid. adopt it to the current structure.

    # Dispatch geometry + backend
    if geom not in ("box", "ellipsoid", "eye", "elliptic_cylinder", "poly"):
        raise ValueError(
            "geom must be 'box' or 'ellipsoid' or 'eye' or 'elliptic_cylinder' or 'poly'"
        )
    if backend not in ("meshpy", "grid"):
        raise ValueError("backend must be 'meshpy' or 'grid'")
    # Prefer meshpy when it is available, unless the caller set force_grid=True


    if geom == "box":
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_box(
                (Lx, Ly, Lz),
                ex,
                ey,
                ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
                knt, ijk = mesh_backend_grid_box(
                    (Lx, Ly, Lz), ex, ey, ez, h=float(h), verbose=bool(verbose)
                )
    elif geom == "ellipsoid":
        # Ellipsoid (now oriented using ex,ey,ez)
        if backend == "meshpy":
            n_subdiv = parse_ell_subdiv_option(
                ell_subdiv, Lx, Ly, Lz, float(h), kappa=1.0
            )
            if (
                verbose
                and isinstance(ell_subdiv, str)
                and ell_subdiv.strip().lower()
                in ("auto", "automatic", "uatomatic", "-1")
            ):
                print(f"[info] auto ell-subdiv = {n_subdiv} for h={h}", flush=True)
            knt, ijk = mesh_backend_meshpy_ellipsoid(
                (Lx, Ly, Lz),
                h=float(h),
                minratio=float(minratio),
                subdiv=int(n_subdiv),
                ex=ex,
                ey=ey,
                ez=ez,
                verbose=bool(verbose),
            )
        else:
                knt, ijk = mesh_backend_grid_ellipsoid(
                    (Lx, Ly, Lz), h=float(h), ex=ex, ey=ey, ez=ez, verbose=bool(verbose)
                )
    elif geom == "eye":
        # Eye: interpret Lx as length, Ly as full width, Lz as thickness
        length = float(Lx)
        width = float(Ly)
        thickness = float(Lz)
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_eye(
                length=length,
                width=width,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_eye(
                length=length,
                width=width,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                verbose=bool(verbose),
            )
    elif geom == "elliptic_cylinder":
        # Elliptic cylinder: cross-section ellipse with semi-axes a=Lx/2, b=Ly/2, extruded along z with thickness Lz
        a = float(Lx) / 2.0
        b = float(Ly) / 2.0
        thickness = float(Lz)
        if backend == "meshpy":
            knt, ijk = mesh_backend_meshpy_elliptic_cylinder(
                a=a,
                b=b,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                minratio=float(minratio),
                verbose=bool(verbose),
            )
        else:
            knt, ijk = mesh_backend_grid_elliptic_cylinder(
                a=a,
                b=b,
                t=thickness,
                ex=ex,
                ey=ey,
                ez=ez,
                h=float(h),
                verbose=bool(verbose),
            )

    elif geom == "poly":
        if isinstance(extent, str):
            Lx, Ly, Lz = parse_csv3(extent)
        else:
            Lx, Ly, Lz = float(extent[0]), float(extent[1]), float(extent[2])
        knt, ijk = mesh_backend_neper_poly(n=int(number_of_grains), 
                                           seed=int(seed), 
                                           size_x=Lx, size_y=Ly, size_z=Lz,
                                           h=float(h))


    # Resolve output filenames
    base = (out_name or "single_solid").strip()
    data_name = out_data_name or base
    vis_name = out_vis_name or base

    # Save data (.npz): knt=(N,3), ijk=(E,5) with mat_id=1
    out_npz = with_ext(data_name, ".npz")
    np.savez(out_npz, knt=knt.astype(np.float64), ijk=ijk.astype(np.int32))
    print(f"[ok] Wrote data: {out_npz} (nodes={knt.shape[0]}, tets={ijk.shape[0]})")

    # Save visualization (.vtu)
    out_vtu: Optional[str] = None
    if not no_vis:
        out_vtu = with_ext(vis_name, ".vtu")
        if not HAVE_meshio:
            print(
                "[warn] meshio not installed; skipping .vtu export. Install with: pip install meshio",
                file=sys.stderr,
            )
            out_vtu = None
        else:
            cells = [("tetra", ijk[:, :4].astype(np.int32))]
            cell_data = {"mat_id": [ijk[:, 4].astype(np.int32)]}
            m = meshio.Mesh(points=knt, cells=cells, cell_data=cell_data)
            m.write(out_vtu)
            print(f"[ok] Wrote visualization: {out_vtu} (cell_data: mat_id)")

    if not return_arrays:
        # Drop large arrays now and return only paths
        del knt, ijk
        return None, None, out_npz, out_vtu
    else:
        return knt, ijk, out_npz, out_vtu




def mesh_backend_neper_poly(n: int, seed: int, size_x: float, size_y: float, size_z: float, h: float):
    import subprocess
    # 1) Generate tessellation
    cmd_tess = ["neper", "-T", "-n", str(n), "-id", str(seed),
                "-morpho", "gg",
                "-morphooptistop", "val=1e-2",
                "-domain", f"cube({size_x},{size_y},{size_z}):translate({-size_x/2},{-size_y/2},{-size_z/2})",
                "-reg", "1"]
    subprocess.run(cmd_tess, check=True)

    # Optional preview (kept as-is)
    cmd_vis = ["neper", "-V", f"n{n}-id{seed}.tess", "-datacellcol", "id", "-print", f"n{n}-id{seed}"]
    subprocess.run(cmd_vis, check=True)

    # 2) Mesh tessellation
    cmd_mesh = ["neper", "-M", f"n{n}-id{seed}.tess", "-cl", f"{h}", "-format", "vtk"]
    subprocess.run(cmd_mesh, check=True)

    # 3) Load VTK and propagate grain IDs
    vtk_path = f"n{n}-id{seed}.vtk"
    mesh = meshio.read(vtk_path)

    knt  = mesh.points
    tets = mesh.cells_dict.get("tetra")
    if tets is None:
        raise RuntimeError("No tetra cells found in Neper output VTK.")

    # Try to find a per-tetra cell-data array to use as material/grain IDs.
    mat = None
    # Prefer the cell_data_dict (present in modern meshio versions)
    try:
        cd_tet = mesh.cell_data_dict.get("tetra", {})
        for key in ("matids", "mat_id", "poly", "grain", "gmsh:physical", "material", "region", "domain"):
            if key in cd_tet:
                mat = np.asarray(cd_tet[key], dtype=np.int32).ravel()
                break
    except Exception:
        pass

    # Fallback: inspect mesh.cell_data (older meshio layout)
    if mat is None and hasattr(mesh, "cell_data"):
        for key, data_list in mesh.cell_data.items():
            # Each data_list aligns with mesh.cells blocks
            for cell_block, data in zip(mesh.cells, data_list):
                if getattr(cell_block, "type", getattr(cell_block, "type", None)) == "tetra":
                    mat = np.asarray(data, dtype=np.int32).ravel()
                    break
            if mat is not None:
                break

    # Last resort: all ones (warn)
    if mat is None:
        print("[warn] No per-tetra cell data found in Neper VTK; defaulting mat_id=1.", file=sys.stderr)
        mat = np.ones((tets.shape[0],), dtype=np.int32)

    # Build ijk (E,5): 4 indices + mat_id
    ijk = np.column_stack([tets, mat])

    return knt, ijk


'''
def mesh_backend_neper_poly(n: int, seed: int, size_x: float, size_y: float, size_z: float, h: float):
    import subprocess
    # 1) Generate tessellation
    cmd_tess = ["neper", "-T", "-n", str(n), "-id", str(seed), 
                "-morpho", "gg",
                "-morphooptistop", "val=1e-2",
                "-domain", f"cube({size_x},{size_y},{size_z}):translate({-size_x/2},{-size_y/2},{-size_z/2})", 
                "-reg", "1"]
    subprocess.run(cmd_tess, check=True)
    cmd_vis = ["neper", "-V", f"n{n}-id{seed}.tess", "-datacellcol", "id","-print", f"n{n}-id{seed}"]
    subprocess.run(cmd_vis, check=True)

    # 2) Mesh tessellation
    cmd_mesh = ["neper", "-M",  f"n{n}-id{seed}.tess",
                "-cl", f"{h}", "-format", "vtk"]
    subprocess.run(cmd_mesh, check=True)

    # 3) Load mesh (VTK) and convert to numpy arrays
    import meshio
    mesh = meshio.read(f"n{n}-id{seed}.vtk")
    knt = mesh.points
    tets = mesh.cells_dict.get("tetra")
    ijk = np.hstack([tets, np.ones((tets.shape[0], 1), dtype=np.int32)])
    return knt, ijk
'''
# ------------------------------- CLI -------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="Single solid mesher (box or ellipsoid) centered at origin with meshpy or grid backend."
    )
    ap.add_argument(
        "--geom",
        type=str,
        default="box",
        choices=["box", "ellipsoid", "eye", "elliptic_cylinder", "poly"],
        help="Select geometry: parallelepiped (box), ellipsoid (symmetry axis is local z), eye, elliptic_cylinder (extruded ellipse), poly (polyhedral grains)",
    )
    ap.add_argument(
        "--extent",
        type=str,
        default="60.0,60.0,60.0",
        help="Full extents Lx,Ly,Lz. Ellipsoid enforces a=b=(Lx+Ly)/2 in local xy-plane.",
    )
    ap.add_argument(
        "--h",
        type=float,
        default=2.0,
        help="Target edge length: meshpy->volume heuristic; grid->brick size ≈ h.",
    )
    ap.add_argument(
        "--minratio",
        type=float,
        default=1.4,
        help="TetGen minradius-to-maxedge ratio (meshpy backend only).",
    )
    ap.add_argument(
        "--backend",
        type=str,
        default="meshpy",
        choices=["meshpy", "grid"],
        help="Choose meshing backend.",
    )

    # Orientation (applies to BOTH box and ellipsoid now)
    ap.add_argument("--dir-x", type=str, default="1,0,0", help="Local x direction.")
    ap.add_argument("--dir-y", type=str, default="0,1,0", help="Local y direction.")
    ap.add_argument(
        "--dir-z",
        type=str,
        default="0,0,1",
        help="Local z direction (ellipsoid symmetry axis).",
    )

    # Ellipsoid surface tessellation (meshpy backend only); allow 'auto'
    ap.add_argument(
        "--ell-subdiv",
        type=str,
        default="auto",
        help="(ELLIPSOID + meshpy) Icosphere subdivision level: "
        "non-negative integer or 'auto'/'automatic'/'-1'.",
    )

    # ap.add_argument(
    #     "--force-grid",
    #     action="store_true",
    #     help="Force use of the grid backend even if meshpy is installed.",
    # )


    ap.add_argument("--n", type=int, default=10, help="Number of grains for polyhedral tessellation")
    ap.add_argument("--id", type=int, default=1, help="Random seed for tessellation")


    # Output naming
    ap.add_argument(
        "--out-name",
        type=str,
        default="single_solid",
        help="Base name for both outputs; adds .npz and .vtu.",
    )
    ap.add_argument(
        "--out-data-name",
        type=str,
        default=None,
        help="Optional override for data filename (adds .npz).",
    )
    ap.add_argument(
        "--out-vis-name",
        type=str,
        default=None,
        help="Optional override for visualization filename (adds .vtu).",
    )

    ap.add_argument(
        "--no-vis", action="store_true", help="Skip writing .vtu visualization file."
    )
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    # Delegate to the programmatic entry point; map CLI types directly.
    try:
        run_single_solid_mesher(
            geom=args.geom,
            extent=args.extent,
            h=float(args.h),
            minratio=float(args.minratio),
            backend=args.backend,
            dir_x=args.dir_x,
            dir_y=args.dir_y,
            dir_z=args.dir_z,
            ell_subdiv=args.ell_subdiv,
            out_name=args.out_name,
            out_data_name=args.out_data_name,
            out_vis_name=args.out_vis_name,
            number_of_grains=args.n,
            seed=args.id,
            no_vis=bool(args.no_vis),
            verbose=bool(args.verbose),
            # CLI uses default (return_arrays=True). For memory-lean CLI, we could add a flag.
            return_arrays=True,
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"[error] Meshing failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
