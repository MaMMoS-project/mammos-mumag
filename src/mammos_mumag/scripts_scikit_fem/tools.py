import configparser
import math

import scipy
from textwrap import dedent

import psutil
import os

import mammos_entity as me
import mammos_mumag
import mammos_units as u

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # in MB

def write_stats(header, total_time, cg_iter, function_calls, hmag_iter):
    print(f'\n--> {header}')
    print('elaplsed time            ', total_time)
    print('iterations               ', cg_iter)
    print('  function calls         ', function_calls)
    # print('  iterations (hmag)      ', hmag_iter)

def write_mh(name, mh):
    me.io.entities_to_file(
        f"{name}.csv",
        dedent(
            f"""\
            Hysteresis loop.
            Generated from mammos_mumag {mammos_mumag.__version__}.
            `configuration_type` is the name of the corresponding vtu file containing the magnetization.
            `B_ext` is the applied magnetic flux density.
            `J` is the polarisation in the direction of the applied magnetic flux.
            `Jx`, `Jy`, `Jz` are the components of the spontaneous polarisation.
            `energy_density` is the energy density."""
        ),
        configuration_type=mh[:,0].astype(int),
        B_ext=me.Entity("MagneticFluxDensity", mh[:,1], u.T),
        J=me.Js(mh[:,2], u.T),
        Jx=me.Js(mh[:,3], u.T),
        Jy=me.Js(mh[:,4], u.T),
        Jz=me.Js(mh[:,5], u.T),
        energy_density=me.Entity("EnergyDensity", mh[:,6] / get_mu0(), u.J / u.m**3),
    )

def get_mu0():
    return scipy.constants.mu_0

def normalize(v):
    a = v[0]
    b = v[1]
    c = v[2]
    s = math.sqrt(a * a + b * b + c * c)
    if s == 0.0:
        return [a, b, c]
    else:
        return [a / s, b / s, c / s]

def read_Js(name):
    with open(name + ".krn") as f:
        ll = f.readline().split()
        Js = float(ll[4])
    return Js

def read_A(name):
    with open(name + ".krn") as f:
        ll = f.readline().split()
        A = float(ll[5])
    return A

def readAnisotropyEnergy(name, m):
    m = normalize(m)
    with open(name + ".krn") as f:
        ll = f.readline().split()
        theta = float(ll[0])
        phi = float(ll[1])
        n0 = math.sin(theta) * math.cos(phi)
        n1 = math.sin(theta) * math.sin(phi)
        n2 = math.cos(theta)
        K1 = float(ll[2])
    mu0 = 4.0e-7 * math.pi
    return -mu0 * K1 * (m[0] * n0 + m[1] * n1 + m[2] * n2) ** 2.0

def read_params(name):
    config = configparser.ConfigParser(
        {
            "mstep": 1.,
            "state": 'mxyz',
            "mfinal": -0.8,
            "hmag_on": 1,
            "tol_fun": 1e-10,
            "tol_hmag_factor": 1.0,
            "precond_iter": 10,
            "iter_max": 1000,
            "verbose": 1,
        }
    )
    config.read(name + ".p2")
    intial_state = config["initial state"]
    field = config["field"]
    minimizer = config["minimizer"]
    m = normalize(
        [
            float(intial_state["mx"]),
            float(intial_state["my"]),
            float(intial_state["mz"]),
        ]
    )
    state = intial_state['state']
    state_id = 0
    if state.lower()=='flower':
      state_id = 1
    if state.lower()=='vortex':
      state_id = 2      
    if state.lower()=='twisted':
      state_id = 3   
    if state.lower()=='random':
      state_id = 4   
    h = normalize([float(field["hx"]), float(field["hy"]), float(field["hz"])])
    hstart, hfinal, hstep = (
        float(field["hstart"]),
        float(field["hfinal"]),
        float(field["hstep"]),
    )
    mstep, mfinal = float(field["mstep"]), float(field["mfinal"])
    hmag_on = int(minimizer["hmag_on"])
    tol_fun = float(minimizer["tol_fun"])
    tol_hmag_factor = float(minimizer["tol_hmag_factor"])
    precond_iter = int(minimizer["precond_iter"])
    iter_max = int(minimizer["iter_max"])
    tol_u = tol_fun * tol_hmag_factor
    verbose = int(minimizer["verbose"])
    # print(f"tolerances: optimality tolerance {tol_fun}   hmag {tol_u}")
    return (                            
        (m,mstep,mfinal,state_id),                                 # magnetic state
        (h,hstart,hfinal,hstep),                                   # field steps
        hmag_on,                                                   # magnetostatics
        (tol_u, tol_fun, precond_iter, iter_max),                  # solver parameters, 
        verbose,                                                   # output
    ) # mag_pars, hext_pars, hmag_on, min_pars, verbose
