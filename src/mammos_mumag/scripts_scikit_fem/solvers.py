import jax
import jax.numpy as np
from jax import jit, lax
from functools import partial

# Inner loop of algorithm 7.1 in nocedal wright
@partial(jit, static_argnums=(0,))
def truncated_cg_diag_precond_jit(Bk, grad_fk, Dk, max_iter=100):
    """
    Solve Bk * p ≈ -grad_fk by preconditioned truncated CG, using a
    diagonal preconditioner from Dk (the diagonal of Bk), and short-circuit
    the iteration as soon as we detect negative curvature or convergence.

    Parameters
    ----------
    Bk : (n, n) array_like
        Symmetric matrix (e.g. approximate Hessian).
    grad_fk : (n,) array_like
        Gradient of f at the current iterate.
    Dk : (n,) array_like
        Diagonal entries of Bk. We assume these are positive for preconditioning.
    max_iter : int
        Maximum number of CG iterations.

    Returns
    -------
    p : (n,) array_like
        The truncated CG step (solution approximation).
    """

    # Tolerance: e_k = min(0.5, sqrt(||grad_fk||) * ||grad_fk||)
    norm_grad = np.linalg.norm(grad_fk)
    epsilon_k = np.minimum(0.5, np.sqrt(norm_grad) * norm_grad)

    # Diagonal preconditioner M^{-1}:  M^{-1} v = v / Dk
    def precond(v):
        return v / Dk  # Assumes Dk[i] != 0 and Dk[i] > 0

    #---------------------------
    # INITIAL STATE
    #---------------------------
    # We'll store:
    #   i:    iteration count
    #   done: boolean indicating we've converged or found negative curvature
    #   p:    current solution approximation
    #   r:    current residual  (r = Bk * p + grad_fk)
    #   z:    preconditioned residual (z = M^{-1} * r)
    #   d:    current search direction
    i0 = np.array(0)
    done0 = np.array(False)

    p0 = np.zeros_like(grad_fk)
    r0 = grad_fk                # Because p0 = 0 => r0 = Bk*0 + grad_fk = grad_fk
    z0 = precond(r0)
    d0 = -z0

    init_state = (i0, done0, p0, r0, z0, d0)

    #---------------------------
    # CONDITION FUNCTION
    #---------------------------
    # The loop continues while i < max_iter AND not done
    def cond_fun(state):
        i, done, p, r, z, d = state
        return (i < max_iter) & (~done)

    #---------------------------
    # BODY FUNCTION
    #---------------------------
    def body_fun(state):
        i, done, p, r, z, d = state

        # d^T (Bk d)
        denom = d @ (Bk(d))

        # Check negative curvature
        negative_curv = (denom <= 0.0)

        # If denom > 0, alpha = (r^T z) / (d^T Bk d), else alpha = 0
        rz = r @ z
        alpha = np.where(~negative_curv, rz / denom, 0.0)

        # Proposed updates
        p_new = p + alpha * d
        r_new = r + alpha * (Bk(d))
        z_new = precond(r_new)

        # Check convergence
        conv = np.linalg.norm(r_new) <= epsilon_k

        # If either negative_curv or conv is true, we are done
        done_new = negative_curv | conv

        #-----------------------------------------
        # Decide final p if we are done this step
        #-----------------------------------------
        # Standard truncated-CG rule:
        #   if negative_curv -> return old p
        #   else if converged -> return new p
        #   else keep going with new p
        p_candidate = np.where(negative_curv, p, p_new)
        p_out       = np.where(~negative_curv & conv, p_new, p_candidate)

        # For r,z,d, if we are done, we won't do another iteration,
        # but to keep the state well-defined, we can "freeze" them:
        r_out = np.where(negative_curv, r, r_new)
        z_out = np.where(negative_curv, z, z_new)

        # Beta = (r_{j+1}^T z_{j+1}) / (r_j^T z_j)
        beta = np.where(
            (~negative_curv & ~conv),
            (r_new @ z_new) / (rz + 1e-16),
            0.0
        )
        d_new = -z_new + beta * d
        d_out = np.where(negative_curv, d, d_new)

        return (i+1, done_new, p_out, r_out, z_out, d_out)

    #---------------------------
    # RUN THE WHILE LOOP
    #---------------------------
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, _, p_final, _, _, _ = final_state

    return lax.cond(
        np.all(p_final == 0),
        lambda _: -grad_fk,
        lambda _: p_final,
        operand=None
    )



@partial(jit, static_argnums=(0,))
def truncated_cg_jit(Bk, grad_fk, max_iter=100):
    """
    Solve Bk * p ≈ -grad_fk by truncated Conjugate Gradient (no preconditioning),
    with short-circuit termination on negative curvature or convergence.

    Parameters
    ----------
    Bk : (n, n) array_like
        Symmetric matrix (e.g. approximate Hessian).
    grad_fk : (n,) array_like
        Gradient of f at the current iterate.
    max_iter : int
        Maximum number of CG iterations.

    Returns
    -------
    p : (n,) array_like
        The truncated CG step (solution approximation).
    """
    # Tolerance: epsilon_k = min(0.5, sqrt(||grad_fk||) * ||grad_fk||)
    norm_grad = np.linalg.norm(grad_fk)
    epsilon_k = np.minimum(0.5, np.sqrt(norm_grad) * norm_grad)

    # Initial state
    #   i: iteration count
    #   done: boolean that indicates we've converged or found negative curvature
    #   p: current solution approximation
    #   r: current residual = Bk*p + grad_fk
    #   d: current search direction
    i0 = np.array(0)
    done0 = np.array(False)

    p0 = np.zeros_like(grad_fk)   # start from 0
    r0 = grad_fk                   # r0 = Bk * p0 + grad_fk = grad_fk
    d0 = -r0                       # initial direction

    init_state = (i0, done0, p0, r0, d0)

    def cond_fun(state):
        i, done, p, r, d = state
        return (i < max_iter) & (~done)

    def body_fun(state):
        i, done, p, r, d = state

        # Negative curvature check
        denom = d @ (Bk(d))
        negative_curv = (denom <= 0.0)

        # alpha = (r^T r) / (d^T Bk d) if denom>0
        rr = r @ r
        alpha = np.where(~negative_curv, rr / (denom + 1e-16), 0.0)

        # Proposed updates
        p_new = p + alpha * d
        r_new = r + alpha * (Bk(d))

        # Check convergence
        conv = np.linalg.norm(r_new) <= epsilon_k

        # If negative curvature or converged => done
        done_new = negative_curv | conv

        # If negative curvature => keep old p, else if converged => new p
        p_candidate = np.where(negative_curv, p, p_new)
        p_out = np.where(~negative_curv & conv, p_new, p_candidate)

        # If we are continuing, compute beta and update d
        # beta = (r_{j+1}^T r_{j+1}) / (r_j^T r_j)
        rr_new = r_new @ r_new
        beta = np.where(~negative_curv & ~conv, rr_new / (rr + 1e-16), 0.0)
        d_new = -r_new + beta * d

        # Freeze the residual/direction if done, so they won't matter
        r_out = np.where(~done_new, r_new, r)
        d_out = np.where(~done_new, d_new, d)

        return (i + 1, done_new, p_out, r_out, d_out)

    # Run the loop
    final_state = lax.while_loop(cond_fun, body_fun, init_state)
    _, _, p_final, _, _ = final_state
    return p_final

    
@jit
def jax_scipy_cg(A, D, b, x0, tol=1e-5, atol=0, maxiter=10000):
    pc = lambda x: x * (1. / D) 
    x, info = jax.scipy.sparse.linalg.cg(A,
                                         b,
                                         x0=x0,
                                         M=pc,
                                         tol=tol,
                                         atol=atol,
                                         maxiter=maxiter)
    return x
