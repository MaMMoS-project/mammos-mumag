# 2D Poisson problem

We take the steady heat conduction problem from [Example 17 in `scikit-fem`](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-17-insulated-wire).

Consider a disk centred in (0,0) having radius 3, and define
```math
\Omega := \{ (x,y) \in \mathbb{R}^2 : \, x^2 + y^2 \le 3 \}.
```

The problem is: find temperature $T$ satisfying
```math
\begin{cases}
    - \nabla \cdot (k \nabla T) = A & \text{in } \Omega, \\
    k \nabla T \cdot \mathbf{n} + hT = 0 \qquad \text{on } \partial \Omega.
\end{cases}
```
where $\mathbf{n}$ is the normal on the surface $\partial \Omega$.

The thermal conductivity is defined as
```math
k(x, y) :=
\begin{cases}
101 & \quad \text{for } 0 \le \sqrt{x^2 + y^2} < 2, \\
11 & \quad \text{for } 2 \le \sqrt{x^2 + y^2} \le 3,
\end{cases}
```
and the Joule heating is
```math
A(x, y) :=
\begin{cases}
5 & \quad \text{for } 0 \le \sqrt{x^2 + y^2} < 2, \\
0 & \quad \text{for } 2 \le \sqrt{x^2 + y^2} \le 3,
\end{cases}
```
and the heat transfer coefficient is set as $h=7$.


We use the Sobolev space $H^1$ as solution space
```math
V := H^1(\Omega) = \{ u \in L^2(\Omega) : \ u' \in L^2(\Omega) \}.
```

The weak formulation reads: find $u \in V$ satisfying
```math
\int_\Omega k(\mathbf{x}) \nabla u(\mathbf{x}) \cdot \nabla v(\mathbf{x}) \mathrm{d} \mathbf{x} + \int_{\partial\Omega} h u(\mathbf{x}) v(\mathbf{x}) \mathrm{d} \sigma = \int_\Omega A(\mathbf{x}) v(\mathbf{x}) \mathrm{d} \mathbf{x} \qquad \forall \, v \in V.
```

## Available notebooks

The notebook [skfem-gmsh.ipynb](./skfem-gmsh.ipynb) contains a solution to the problem coming from [Example 17 in `scikit-fem`](https://scikit-fem.readthedocs.io/en/latest/listofexamples.html#example-17-insulated-wire) using an external mesh (`disk.msh`) generated using `gmsh`.

The notebook [skfem-salome.ipynb](./skfem-salome.ipynb) contains the same problem using an external mesh (`disk.med`) generated using [Salome](https://www.salome-platform.org/?page_id=374) and the generating script `disk.py`.
