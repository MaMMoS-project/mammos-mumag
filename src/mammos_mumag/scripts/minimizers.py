import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as np
from jax import jit, lax
from jax.experimental import sparse
from jax.scipy.optimize import minimize
from functools import partial
import scipy.sparse as sp
import sys
from time import time

from jax_tools import update_x
from solvers import jax_scipy_cg

# Finite difference interval of algorithm 2 in AIP Advances 7, 045310 (2017)
@jit
def compute_h(x, d):
    eps_m = np.finfo(x.dtype).eps
    sqrt_eps_m = np.sqrt(eps_m)
    
    norm_x_2 = np.linalg.norm(x, ord=2)
    norm_d_2 = np.linalg.norm(d, ord=2)
    norm_d_inf = np.linalg.norm(d, ord=np.inf)

    term1 = 2.0 * sqrt_eps_m * (1.0 + norm_x_2) / norm_d_2
    term2 = sqrt_eps_m / norm_d_inf

    return lax.cond(term1 < term2,
                    lambda _: term1,
                    lambda _: term2,
                    operand=None)

# Initial step length of algorithm 2 in AIP Advances 7, 045310 (2017)
@jit
def alpha_init(f0,      # energy of previous iteration
               f,g,d,   # energy, gradient, search direction
               g1,      # gradient at x + h*d
               h,       # finite difference step siz
               c=0.1,   # step length scale factor for negative curvature
               o=0.99   # o < 1 takes more often alpha_0 from nocedal, wright eq 3.60
               ):   
    gd = np.dot(g,d)
    term1 = -h*(gd/(np.dot(g1,d)-gd))
    term1 = lax.cond(term1 < 0,
                 lambda t: c * np.abs(t),  # negative curvature
                 lambda t: t,
                 term1)
    term2 = 2.0*(f-f0)/gd
    term2 = lax.cond(term2 < 0,
                 lambda t: 1.0e30,        # use high value, select term1 afterwards
                 lambda t: t,
                 term2)
    # jax.debug.print('initial step length {a} {b} {x}',a=term1,b=term2,x=term1<o*term2)
    return lax.cond(term1 < o*term2,
                    lambda _: term1,
                    lambda _: term2,
                    operand=None)

'''
o = 1
iterations                1196
  function calls          2553
  
o = 0.99
iterations                1192
  function calls          2545

o = 0.98
iterations                1215
  function calls          2591

o = 0.95
iterations                1264
  function calls          2689

o = 0.9  
iterations                1262
  function calls          2685  
  
o = 1.01
iterations                1203
  function calls          2567


o = 1.10
--> done
iterations                1199
  function calls          2559

'''

@partial(jit, static_argnums=(5,9,))
def line_search(x,      # current magnetization
                f0,     # energy of previous interation
                f,g,d,  # energy, gradient, search direction  
                func, func_args, alt_args, stats,
                update_x):
    
    rho=0.5
    c=1e-4
    delta=0.1
    eps=1e-6                                                          # see section 5 of hager zhang paper  
    max_iter=20
    
    h  = compute_h(x, d)

    x1 = update_x(x, h, d)
    f1, g1, alt_args, stats = func(x1, func_args, alt_args, stats)
    alpha = alpha_init(f0,f,g,d,g1,h)   
                       
    gd = np.dot(g, d)
    
    def cond(state):
        i, gd1, f1, alpha, _, _, _, _ = state
        eps_k = eps*np.abs(f)                                         # eq 4.3 in hager zhang paper 
        wolfe = np.logical_and((2*delta-1)*gd >= gd1,f1 <= f + eps_k) # approximate Wolfe  SIAM J. OPTIM. Vol. 16, No. 1, pp. 170–192
        done  = np.logical_or(wolfe, f1 <= f + c * alpha * gd)        # or armijo 
        done  = np.logical_or(done, i > max_iter)   # maximum iterations exceeded
        return np.logical_not(done)
  
    def body(state):
        i, _, _, alpha, _, _, alt_args, stats = state
        alpha *= rho
        jax.debug.print('STEP SIZE DECREASED to {alpha} {i}',alpha=alpha, i=i)
        x1 = update_x(x, alpha, d)
        f1, g1, alt_args, stats = func(x1, func_args, alt_args, stats)
        gd1 = np.dot(g1, d)         
        return i+1, gd1, f1, alpha, x1, g1, alt_args, stats
  
    x1 = update_x(x, alpha, d)
    f1, g1, alt_args, stats = func(x1, func_args, alt_args, stats)
    gd1 = np.dot(g1, d)         
    state = 0, gd1, f1, alpha, x1, g1, alt_args, stats
    
    _, gd1, f1, alpha, x1, g1, alt_args, stats = lax.while_loop(cond, body, state)
    
    return x1, f1, g1, alt_args, stats

# Algorithm 4 Computer Physics Communications 235 (2019) 179–186
@partial(jit, static_argnums=(1,6,7,))
def hestenes_stiefel_ncg(x, func, func_args, alt_args, stats, tol, update_x, M_inv):
    iter_max = 100000
    
    f0 = np.finfo(x.dtype).max
    f, g, alt_args, stats = func(x, func_args, alt_args, stats)
    d = -M_inv(x, g, func_args, alt_args) # Initial search direction
    k = 1
    n_restart = 20

    def print_restart_z1(_):
        jax.debug.print("restart: -z1 is not a sufficient downhill direction")
        return None

    def print_restart_d(_):
        jax.debug.print("restart:  d  is not a sufficient downhill direction")
        return None

    def no_op(_):
        return None

    def cond(state):
        k, x0, f0, x, f, g, _, _, _ = state
        # jax.debug.print('    min   {k} {f}',k=k,f=f)
        a = (f0-f) > tol*(1+np.abs(f))                # gill, murray, wright, practical optimization, section 8.2.3.2
        b = np.logical_or(np.linalg.norm((x0-x),ord=np.inf) > np.sqrt(tol)*(1+np.linalg.norm(x,ord=np.inf)),k==0)
        c = np.linalg.norm(g,ord=np.inf) > np.cbrt(tol)*(1+np.abs(f))
        return np.logical_and(np.logical_or(a, np.logical_or(b, c)),k < iter_max)

    def body(state):
        k, _, f0, x, f, g, d, alt_args, stats = state
        condition = np.dot(d,g) > -0.001*np.linalg.norm(d)*np.linalg.norm(g)   # eq 2.15, Andrei, Open Problems in Nonlinear Conjugate Gradient ...
        jax.lax.cond(condition, print_restart_z1, no_op, operand=None)
        d = jax.lax.select(condition, -g, d)                                   # use -g, if d not downhill
        x1, f1, g1, alt_args, stats = line_search(x,f0, f,g,d, func, func_args, alt_args, stats, update_x) 
        z1 = M_inv(x, g1, func_args, alt_args)
        y = g1 - g
        beta = np.maximum(np.dot(y,z1) / np.dot(y,d), 0.)  
                                                          # HS+, Hager, Zhang, A SURVEY OF NONLINEAR CONJUGATE GRADIENT METHODS
        # condition = (k % n_restart) == 0
        # beta = jax.lax.select(condition, 0.0, beta)
        d = -z1 + beta * d
        condition = np.dot(d,g) > -0.001*np.linalg.norm(d)*np.linalg.norm(g)   # eq 2.15, Andrei, Open Problems in Nonlinear Conjugate Gradient ...
        jax.lax.cond(condition, print_restart_d, no_op, operand=None)
        d = jax.lax.select(condition, -z1, d)                                  # use -z1, if d not downhill
        return k+1, x, f, x1, f1, g1, d, alt_args, stats

    state = k, x, f0, x, f, g, d, alt_args, stats
    k, _, _, x1, f1, g1, _, alt_args, stats = lax.while_loop(cond, body, state)
                    
    return f1, x1, k, alt_args, stats

# Example usage:
if __name__ == '__main__':
    
    # parameters for minimizer
    tol      = 1e-10     # tolerance for stopping
    precond_iter = 20    # cg step for preconditioning
    iter_max = 1000      
        
    # Define a simple quadratic function:
    # f(x) = 0.5 * x^T A x - b^T x, where A is positive-definite.
    n = 2000
    A = sp.random(n, n, density=0.2, format='csr')  # constant part of A(x)
    A = A.T @ A + sp.eye(n)                         # make it positive definite
    D = np.array(A.diagonal())
    A = sparse.BCOO.from_scipy_sparse(A).sort_indices()
    key = jax.random.PRNGKey(42)
    b   = jax.random.uniform(key, shape=(n,))
    
    @jit
    def M_inv(x,g,func_args,alt_args):
       A, D, _, precond_iter = func_args
       norm_g = np.linalg.norm(g)
       atol = np.minimum(0.5,np.sqrt(norm_g))*norm_g         # nocedal, wright, algorithm 7.1
       return jax_scipy_cg(A, D, g, np.zeros(len(g)), tol=0, atol=atol, maxiter=precond_iter)

    @jit
    def func(x,func_args,alt_args,stats):
        A, D, b, _ = func_args
        calls = stats
        Ax = A @ x
        e  = 0.5 * np.dot(x, Ax) - np.dot(b, x)
        g  = Ax - b
        return e, g, alt_args, calls+1
    
    # Initial guess
    x0 = np.zeros(n)
    alt_args = np.zeros(1)  # dummy array
    
    func_time = 0
    calls     = 0
    stats = calls
    f_hs, x_hs, iterations_hs, _, _ = hestenes_stiefel_ncg(x0, func, (A,D,b,precond_iter,), (alt_args,), stats, tol, update_x, M_inv)
                                                        
    t0 = time()
    func_time = 0
    calls    = 0
    stats = calls
    f_hs, x_hs, iterations_hs, _, stats = hestenes_stiefel_ncg(x0, func, (A,D,b,precond_iter,), (alt_args,), stats, tol, update_x, M_inv)
    elapsed_time = time()-t0
    func_calls = stats
    
    print("\n")
    print("Hestenes-Stiefel NCG")
    print("total time      :", elapsed_time)
    print("function value  :", f_hs)
    print("function calls  :", func_calls)
    print("iterations      :", iterations_hs)


    
