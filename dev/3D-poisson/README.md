# 3D Poisson problem

We expand [the problem defined in 2D](../2D-poisson/README.md) in 3D.

Consider a sphere centred in (0,0,0) having radius 3, and define
```math
\Omega := \{ \mathbf{x} = (x,y,z) \in \mathbb{R}^3 : \, \sqrt{x^2 + y^2 + y^3} \le 3 \}.
```

The problem is: find $u$ satisfying
```math
\begin{cases}
    - \nabla \cdot (k \nabla u) = A & \text{in } \Omega, \\
    k \nabla u \cdot \mathbf{n} + h u = 0 \qquad \text{on } \partial \Omega.
\end{cases}
```
where $\mathbf{n}$ is the normal on the surface $\partial \Omega$.

The parameters are defined as
```math
k(x, y, z) :=
\begin{cases}
101 & \quad \text{for } 0 \le \sqrt{x^2 + y^2 + z^2} < 2, \\
11 & \quad \text{for } 2 \le \sqrt{x^2 + y^2 + z^2} \le 3,
\end{cases}
```
```math
A(x, y, z) :=
\begin{cases}
5 & \quad \text{for } 0 \le \sqrt{x^2 + y^2 + z^2} < 2, \\
0 & \quad \text{for } 2 \le \sqrt{x^2 + y^2 + z^2} \le 3,
\end{cases}
```
and $h=7$.


We use the Sobolev space $H^1(\Omega)$ as solution space
```math
V := H^1(\Omega) = \{ u \in L^2(\Omega) : \ u' \in L^2(\Omega) \}.
```

The weak formulation reads: find $u \in V$ satisfying
```math
\int_\Omega k(\mathbf{x}) \nabla u(\mathbf{x}) \cdot \nabla v(\mathbf{x}) \mathrm{d} \mathbf{x} + \int_{\partial\Omega} h u(\mathbf{x}) v(\mathbf{x}) \mathrm{d} \sigma = \int_\Omega A(\mathbf{x}) v(\mathbf{x}) \mathrm{d} \mathbf{x} \qquad \forall \, v \in V.
```

<!-- ## Available notebooks

The notebook [skfem-gmsh.ipynb](./skfem-gmsh.ipynb) contains a solution to the problem using an external mesh (`disk.msh`) generated using `gmsh`.

The notebook [skfem-salome.ipynb](./skfem-salome.ipynb) contains the same problem using an external mesh (`disk.med`) generated using [Salome](https://www.salome-platform.org/?page_id=374) and the generating script `disk.py`. -->
