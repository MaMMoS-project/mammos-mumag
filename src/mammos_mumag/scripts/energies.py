import sys
import jax.numpy as np
from jax import jit, lax
import jax
from solvers import jax_scipy_cg

@jit
def external_g(value,params):
    direction,meas,volume = params
    return np.concatenate([(-value*direction[0]/volume)*meas,
                           (-value*direction[1]/volume)*meas,
                           (-value*direction[2]/volume)*meas]).reshape(3,-1).T.flatten()
    
@jit
def external_e(m,value,params):
    return np.dot(m, external_g(value,params))
    
@jit
def external_eg(m,value,params):
    g = external_g(value,params)
    return np.dot(m, g), g
@jit
def exani_g(m, mat_exani):
    return mat_exani @ m

@jit
def exani_e(m, mat_exani):
    return 0.5*np.dot(m,mat_exani@m)

@jit
def exani_eg(m, mat_exani):
    g = mat_exani@m
    return 0.5*np.dot(m,g), g


@jit
def hmag_g(m, u0, tol, params):
    dx, dy, dz, A, D, gx, gy, gz = params

    #def Afunc(x):
    #    return A @ x

    b = dx@m[0::3] + dy@m[1::3] + dz@m[2::3] 
    u = jax_scipy_cg(A, D, b, u0, tol)
    g = np.concatenate([gx@u,gy@u,gz@u]).reshape(3,-1).T.flatten()
    return g, u

@jit
def hmag_e(m, u0, tol, params):
    g, u = hmag_g(m, u0, tol, params)
    return 0.5 * np.dot(m, g), u
    
@jit
def hmag_eg(m, u0, tol, params):
    #jax.debug.print('u in  hmag {out}',out=u0[4729])
    #jax.debug.print('tol_u {tol}',tol=tol)
    g, u = hmag_g(m, u0, tol, params)
    #jax.debug.print('u out hmag {out}',out=u[4729])
    return 0.5 * np.dot(m, g), g, u

@jit
def projection(m, g):  # eq (15) Computer Physics Communications 235 (2019) 179–186
    m_ = np.reshape(m,(-1,3))
    g_ = np.reshape(g,(-1,3))
    return np.cross(m_, np.cross(g_, m_)).flatten()
    
@jit
def total_eg(m, args, other, stats):
    u0, _ = other
    tol, field_value, pars = args
    calls = stats
    external_pars = pars['hext_pars'][0], pars['meas'], pars['volume']
    energy, gradient = external_eg(m,field_value,external_pars)
    
    exani_A = pars['exani_pars'][0] 
    exani_D = pars['exani_pars'][1] 
    e, g_exani = exani_eg(m,exani_A)
    energy    += e
    gradient  += g_exani 
    
    def hmag_true(_):
        return hmag_eg(m, u0, tol, pars['hmag_pars'])
    
    def hmag_false(_):
        zero_energy = 0.0
        zero_gradient = np.zeros_like(gradient)  
        return zero_energy, zero_gradient, u0

    e_hmag, g_hmag, u = lax.cond(
        pars['hmag_on'] > 0,
        hmag_true,
        hmag_false,
        operand=None
    )
    
    energy   += e_hmag
    gradient += g_hmag

    return energy, projection(m,gradient), (u, gradient), calls+1  
    
if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript energies.py modelname")
