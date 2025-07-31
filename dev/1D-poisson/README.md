# Problem definition

Consider the interval domain $\Omega = [0, 1]$. We consider the problem
$$-(\sigma(x) u'(x) )' = 0$$
with Dirichlet boundary conditions $u(0)=0$ and $u(1)=10$.

The function $\sigma$ is given as
```math
\sigma(x) :=
\begin{cases}
1 & \quad \text{for } x \in [0, \frac{1}{2}), \\
2 & \quad \text{for } x \in [\frac{1}{2}, 1].
\end{cases}
```

Defining the function space
```math
V = \{ v \in H^1(\Omega) : \ v(0)=0, \ v(1)=10 \},
```
where $H^1(\Omega)$ is the Sobolev space
```math
H^1(\Omega) = W^{1,1} (\Omega) := \{ u \in L^2(\Omega) : \ u' \in L^2(\Omega) \}.
```

The weak formulation reads: find $u \in V$ satisfying
```math
\int_\Omega \sigma(x) u'(x) v'(x) \mathrm{d}x = 0  \qquad \forall \, v \in V.
```

The analytical solution is
```math
u(x)=
\begin{cases}
\frac{40}{3}x & \quad \text{for } x \in [0, \frac{1}{2}), \\
\frac{20}{3}x + \frac{10}{3} & \quad \text{for } x \in [\frac{1}{2}, 1].
\end{cases}
```


## Available notebooks

1. [skfem-simple.ipynb](./skfem-simple/skfem-simple.ipynb) <br>
   Evaluate the solution using [`scikit-fem`](https://github.com/kinnala/scikit-fem). The mesh is created with the functions found in `scikit-fem`.
2. [fenics-simple.ipynb](./fenics-simple/fenics-simple.ipynb) <br>
    Evaluate the solution using [`FEniCSx`](https://fenicsproject.org/). The mesh is created with the functions found in `FEniCSx`.
3. [fenics-gmsh.ipynb](./fenics-gmsh/fenics-gmsh.ipynb) <br>
    Evaluate the solution using `FEniCSx`. The mesh is first defined using [`gmsh`](https://gmsh.info/) and then loaded using functions found in FEniCSx.
4. [skfem-gmsh.ipynb](./skfem-gmsh/skfem-gmsh.ipynb) <br>
    Evaluate the solution using `scikit-fem`. The mesh is first defined using `gmsh` and then loaded using functions found in `scikit-fem`.
   
