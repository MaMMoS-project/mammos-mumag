import inspect
import jax.numpy as np
import math
import sys
from time import time

import esys.escript as e

from converters import operator2matrix, escript2numpy, escript2jax, csr2bcoo, toEscriptScalar
from energies import external_eg, exani_eg, hmag_eg, total_eg
from magnetization import xM, getM
from matrix import exani_matrix, dx, dy, dz, gx, gy, gz, stiffness_matrix
from materials import Materials
from tools import read_params, read_Js, read_A, readAnisotropyEnergy, write_stats, get_memory_usage, get_mu0
from jax_tools import normalize_vectors

import gc

total_energy = 0

# test of external field computation
def check_external(name,m_np,pars,out=True):
    Js = read_Js(name)
    m, _, _, _ = pars['mag_pars']
    h, start, final, step = pars['hext_pars']
    field_value = (start - step)
    ezee = -Js * field_value * (m[0] * h[0] + m[1] * h[1] + m[2] * h[2])
    external_pars = pars['hext_pars'][0], pars['meas'], pars['volume']
    energy, _ = external_eg(m_np,field_value,external_pars)
    if out:
        mu0 = get_mu0()
        with open(name + "_zeeman.csv", "w") as file:
            file.write(
                inspect.cleandoc(
                    f"""
                    Zeeman energy density for an external field of mu_0 Hext {field_value} (T).
                    name,value,explanation
                    E_jax,{energy/mu0},Energy evaluated with jax backend (J/m^3).
                    E_analytic,{ezee/mu0},Energy evaluated analytically (J/m^3).
                    """
                )
            )
    return energy

# test exchange and anisotropy
def check_exchange(name,m_np,pars,meas,out=True):
    scale = pars['volume']**0.3333333333333333
    size = pars['size']
    m, k = xM(e.wherePositive(meas), scale)
    A = read_A(name)
    mu0 = 4.0e-7 * math.pi
    exani_A = pars['exani_pars'][0] 
    m_np_x = escript2jax(m)

    energy, _ = exani_eg(m_np_x,exani_A)

    if out:
        mu0 = get_mu0()
        with open(name + "_exchange.csv", "w") as file:
            file.write(
                inspect.cleandoc(
                    f"""
                    Exchange energy density of a vortex on a, b plane.
                    name,value,explanation
                    E_jax,{energy/mu0},Energy evaluated with jax backend (J/m^3).
                    E_analytic,{(2 * k * k * mu0 * A / (size * size))/mu0},Energy evaluated analytically (J/m^3).
                    """
                )
            )
    return energy

def check_anisotropy(name,m_np,pars,out=True):
    global total_energy
    uniform_m = [0.0, 0.0, 1.0]
    eani = readAnisotropyEnergy(name, uniform_m)
    exani_A = pars['exani_pars'][0]
    energy, _ = exani_eg(m_np,exani_A)

    if out:
        mu0 = get_mu0()
        with open(name + "_anisotropy.csv", "w") as file:
            file.write(
                inspect.cleandoc(
                    f"""
                    Magnetocrystalline anisotropy energy density of uniformly magnetized sample in direction {uniform_m}.
                    name,value,explanation
                    E_jax,{energy/mu0},Energy evaluated with jax backend (J/m^3).
                    E_analytic,{eani/mu0},Energy evaluated analytically (J/m^3).
                    """
                )
            )
        total_energy += energy/mu0

    return energy

def check_exani(name,m_np,pars,meas,out=True):
    check_exchange(name,m_np,pars,meas,out)
    return check_anisotropy(name,m_np,pars,out)

# test magnetostatics
def check_hmag(name,m,pars,out=True):   
    Js  = read_Js(name)
    tol = 1e-10
    energy, _, _ = hmag_eg(m,np.zeros(len(m)//3),tol,pars['hmag_pars'])
    if out:
        mu0 = get_mu0()
        with open(name + "_hmag.csv", "w") as file:
            file.write(
                inspect.cleandoc(
                    f"""
                    Magnetostatic energy density of uniformly magnetized cube.
                    name,value,explanation
                    E_jax,{energy/mu0},Energy evaluated with jax backend (J/m^3).
                    E_analytic,{(Js * Js / 6)/mu0},Energy evaluated analytically (J/m^3).
                    """
                )
            )
    return energy    
        
'''
mag_pars
hext_pars
hmag_on
min_pars
verbose
meas
volume
size
exani_pars
hmag_pars
'''
def pars2jax(pars):
    pars['meas'] = np.array(pars['meas'])
    
    C, D = pars['exani_pars']
    C = csr2bcoo(C)
    D = np.array(D)
    pars['exani_pars'] = C, D
    
    mat_dx, mat_dy, mat_dz, mat_stiff, dia_stiff, mat_gx, mat_gy, mat_gz = pars['hmag_pars']
    mat_dx = csr2bcoo(mat_dx)
    mat_dy = csr2bcoo(mat_dy)
    mat_dz = csr2bcoo(mat_dz)
    mat_stiff = csr2bcoo(mat_stiff)
    dia_stiff = np.array(dia_stiff)
    mat_gx = csr2bcoo(mat_gx)
    mat_gy = csr2bcoo(mat_gy)
    mat_gz = csr2bcoo(mat_gz)
    pars['hmag_pars'] = mat_dx, mat_dy, mat_dz, mat_stiff, dia_stiff, mat_gx, mat_gy, mat_gz 

# update pars from file
def update_pars(name,pars):
    mag_pars, hext_pars, hmag_on, min_pars, verbose  = read_params(name)
    pars['mag_pars']  = mag_pars
    pars['hext_pars'] = hext_pars
    pars['hmag_on']   = hmag_on
    pars['min_pars']  = min_pars
    pars['verbose']   = verbose
        
  
# update initial magnetization
def initial_m(domain, pars):
    mag_pars = pars['mag_pars']
    meas     = toEscriptScalar(pars['meas'],domain)
    m, _, _, state_id = mag_pars
    m_e  = getM(e.wherePositive(meas), m, state_id)
    return escript2jax(m_e)
  
  
# mapping between fem tool and sparse matrices
def escript2arrays(name,check=0,target='jax'):
    global total_energy
    
    pars = {}
    
    # parameters
    mag_pars, hext_pars, hmag_on, min_pars, verbose  = read_params(name)
    m, _, _, state_id = mag_pars
    h, _, _, _        = hext_pars
    pars['mag_pars']  = mag_pars
    pars['hext_pars'] = hext_pars
    pars['hmag_on']   = hmag_on
    pars['min_pars']  = min_pars
    pars['verbose']   = verbose
    
    # materials
    materials = Materials(name)
    if target=='jax':
      pars['meas'] = escript2jax(materials.meas)
    else:
      pars['meas'] = escript2numpy(materials.meas)
    pars['volume'] = materials.volume
    pars['size']   = materials.size
    
    # initial magnetization
    m_e  = getM(e.wherePositive(materials.meas), m, state_id)
    if target=='jax':
      m_np = escript2jax(m_e)
    else:
      m_np = escript2numpy(m_e)
            
    # exchange and anisotropy
    if target=='jax':
        C = operator2matrix(exani_matrix(materials.A, materials.K, materials.u, materials.volume),diag=False)[0]
    else:
        C = operator2matrix(exani_matrix(materials.A, materials.K, materials.u, materials.volume),fm='csr',diag=False)[0]      
    D = operator2matrix(exani_matrix(materials.A, None, materials.u, materials.volume),fm='csr')[1]
    m3 = np.concatenate([pars['meas'],pars['meas'],pars['meas']]).reshape(3,-1).T.flatten()
    D = np.where(m3 == 0, 1, D)
    pars['exani_pars'] = C, D
    
    # magnetostatic
    if target=='jax':
        mat_dx = operator2matrix(dx(materials.Js),diag=False)[0] 
        mat_dy = operator2matrix(dy(materials.Js),diag=False)[0] 
        mat_dz = operator2matrix(dz(materials.Js),diag=False)[0] 
        mat_stiff, dia_stiff = operator2matrix( stiffness_matrix(materials.Js, materials.volume) )
        mat_gx = operator2matrix(gx(materials.Js, materials.volume),diag=False)[0]
        mat_gy = operator2matrix(gy(materials.Js, materials.volume),diag=False)[0]
        mat_gz = operator2matrix(gz(materials.Js, materials.volume),diag=False)[0]
    else:
        mat_dx = operator2matrix(dx(materials.Js),fm='csr',diag=False)[0] 
        mat_dy = operator2matrix(dy(materials.Js),fm='csr',diag=False)[0] 
        mat_dz = operator2matrix(dz(materials.Js),fm='csr',diag=False)[0] 
        mat_stiff, dia_stiff = operator2matrix( stiffness_matrix(materials.Js, materials.volume), fm='csr' )
        mat_gx = operator2matrix(gx(materials.Js, materials.volume),fm='csr',diag=False)[0]
        mat_gy = operator2matrix(gy(materials.Js, materials.volume),fm='csr',diag=False)[0]
        mat_gz = operator2matrix(gz(materials.Js, materials.volume),fm='csr',diag=False)[0]
    pars['hmag_pars'] = mat_dx, mat_dy, mat_dz, mat_stiff, dia_stiff, mat_gx, mat_gy, mat_gz       
                      
    for i in range(check):
      total_energy += check_external(name,m_np,pars,i==0)  
      total_energy += check_exani(name,m_np,pars,materials.meas,i==0)
      total_energy += check_hmag(name,m_np,pars,i==0)
        
    return m_np, pars, materials.get_tags()
        
if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript mapping.py modelname [store]")
    
    store = 0
    
    memory_pre = get_memory_usage()
    m, pars, _ = escript2arrays(name,1)
    memory_post = get_memory_usage()
    gc.collect()
    memory_collected = get_memory_usage()

    u0 = np.zeros(len(m)//3)
    h, start, final, step = pars['hext_pars']
    field_value = (start - step)
    tol_u = pars['min_pars'][0]

    args = tol_u, field_value, pars    
    
    stats = 0
    t0 = time()
    energy, _, _, stats = total_eg(m, args, (u0,None,), stats)
    function_calls = stats
    total_time = time() - t0
        
    stats = 0
    t0 = time()
    energy, _, _, stats = total_eg(m, args, (u0,None,), stats)
    function_calls = stats

    with open(name + "_stats.txt", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                MAP FINITE ELEMENT BACKEND (esys-escript) TO JAX.
                Memory before escript2jax: {memory_pre} MB.
                Memory after  escript2jax: {memory_post} MB.
                Memory after garbage collection: {memory_collected} MB.
                Timing and statistics.
                elapsed time: {total_time}
                function_calls: {function_calls}
                """
            )
        )
                           
    with open(name + "_energy.csv", "w") as file:
        file.write(
            inspect.cleandoc(
                f"""
                Total energy density of uniformly magnetized state.
                name,value,explanation
                E_jax,{energy/get_mu0()},Energy evaluated with jax backend (J/m^3).
                E_analytic,{total_energy/get_mu0()},Energy evaluated analytically (J/m^3).
                """
            )
        )
