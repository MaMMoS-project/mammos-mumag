import inspect
import jax
jax.config.update("jax_enable_x64", True)
import sys
from time import time
from functools import partial
import os

import jax.numpy as np
from jax import jit, lax
import numpy

from energies import total_eg, projection
from mapping import escript2arrays, update_pars, initial_m
from minimizers import hestenes_stiefel_ncg
from solvers import truncated_cg_diag_precond_jit, truncated_cg_jit, jax_scipy_cg
from store import pickle2jax
from tools import write_mh, write_stats, get_memory_usage
from jax_tools import update_m, dot_magnetizations

from escript_tools import write_magnetization_and_potential, readmesh_get_tags

import gc

@jit
def compute_mh(m, pars):
    direction, meas, volume = pars
    mh =  (
        np.dot(
            (
                  direction[0] * m[0::3]
                + direction[1] * m[1::3]
                + direction[2] * m[2::3]
            ),
            meas,
        )
        / volume
    )
    mx = np.dot(m[0::3],meas)/volume
    my = np.dot(m[1::3],meas)/volume
    mz = np.dot(m[2::3],meas)/volume
    return mh, mx,my,mz

@jit
def M_inv(x,g,func_args,alt_args):
   tol, field_value, pars = func_args
   C = pars['exani_pars'][0]
   D = pars['exani_pars'][1]
   #D = np.ones(len(x))
   _, gradF = alt_args
   min_pars = pars['min_pars']
   precond_iter =  min_pars[2]

   def A(v):                                         # eq (22) Computer Physics Communications 235 (2019) 179–186
       r1 = projection(m,C@v)
       r2 = dot_magnetizations(m,gradF)*v.reshape(-1,3)
       return r1 - r2.flatten()

   return truncated_cg_diag_precond_jit(A,-g,D,max_iter=precond_iter)


@jit
def minimize(m, field_value, pars, alt_args, stats):
  min_pars = pars['min_pars']
  tol_u = min_pars[0]
  tol_fun = min_pars[1]
  args = tol_u, field_value, pars
  energy, m, cg_iter, alt_args, stats = hestenes_stiefel_ncg(m, total_eg, args, alt_args, stats, tol_fun, update_m, M_inv)
  return energy, m, cg_iter, alt_args, stats

def save_callback(m,u,counter):
    write_magnetization_and_potential(name,counter+1,m,u,tags)
    return counter + 1

@partial(jit, static_argnums=(0,3,))
def solve(name, m0, pars, max_steps):
    mfinal = pars['mag_pars'][2]
    mstep  = pars['mag_pars'][-3]
    hdir   = pars['hext_pars'][0]
    hstart = pars['hext_pars'][1]
    hstep  = pars['hext_pars'][3]

    stats = 0
    mh = np.finfo(m0.dtype).max

    rec = np.zeros((max_steps, 7))
    u0 = np.zeros(len(m0)//3)
    gradF = np.zeros(len(m0))
    alt_args = (u0, gradF)

    save_counter = 0
    last_saved_mh = mh

    def cond(state):
        _, mh, _, _, _, _, _, idx, _, _ = state
        return np.logical_and(
            mh > mfinal,
            idx < max_steps
        )

    def body(state):
        m, mh, hext, cum_iter, alt_args, stats, rec, idx, counter, last_saved_mh  = state
        energy, m, cg_iter, alt_args, stats = minimize(m, hext, pars, alt_args, stats)
        u, _ = alt_args
        mh, mx,my,mz = compute_mh(m, (hdir, pars['meas'], pars['volume']))
        # jax.debug.print("--> demag {hext} {mh}", hext=hext, mh=mh)
        should_save = (last_saved_mh - mh) > mstep

        def true_fun(_):
            new_counter = jax.experimental.io_callback(
                save_callback,
                jax.ShapeDtypeStruct((), np.int64),
                m, u, counter
            )
            return new_counter, mh

        def false_fun(_):
            return counter, last_saved_mh

        counter, last_saved_mh = lax.cond(should_save, true_fun, false_fun, operand=None)
        rec = rec.at[idx].set(np.array([counter, hext, mh, mx,my,mz, energy]))

        return m, mh, hext+hstep, cum_iter+cg_iter, alt_args, stats, rec, idx+1, counter, last_saved_mh

    state = m0, mh, hstart, 0, alt_args, stats, rec, 0, save_counter, last_saved_mh
    _, _, _, cum_iter, _, stats, rec, idx, _, _ = lax.while_loop(cond, body, state)

    return cum_iter, stats, rec, idx

def loop(name,m,pars):
    # print(pars)
    hstart = pars['hext_pars'][1]
    hfinal = pars['hext_pars'][2]
    hstep  = pars['hext_pars'][3]
    max_iter = int(abs(hfinal-hstart)/abs(hstep))+1

    t0 = time()
    cg_iter, stats, rec, idx = solve(name,m,pars,max_iter)
    jax.block_until_ready(rec[1])  # block until mh is done
    jax.block_until_ready(rec[2])  # block until mx is done
    jax.block_until_ready(rec[3])  # block until my is done
    jax.block_until_ready(rec[4])  # block until mz is done

    total_time  = time()-t0

    hmag_iter = 0

    function_calls = stats
    #write_stats('done',total_time,cg_iter,function_calls,hmag_iter)

    write_mh(name,rec[:idx])

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript loop.py modelname")

    # jax.profiler.start_trace("/tmp/tensorboard")

    memory_pre = get_memory_usage()
    if os.path.exists(f'{name}.pkl'):
        print('read stored matrices')
        m, pars = pickle2jax(name)
        tags    = readmesh_get_tags(name)
        update_pars(name, pars)
        m = initial_m(tags.getDomain(),pars)
    else:
        m, pars, tags = escript2arrays(name,0)
    memory_post = get_memory_usage()
    gc.collect()
    memory_collected = get_memory_usage()

    with open(name + "_stats.txt", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Memory before escript2jax: {memory_pre} MB.
                Memory after  escript2jax: {memory_post} MB.
                Memory after garbage collection: {memory_collected} MB.
                """
            )
        )

    loop(name,m,pars)

    # jax.profiler.stop_trace()
