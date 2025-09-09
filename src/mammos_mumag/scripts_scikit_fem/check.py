import jax
jax.config.update("jax_enable_x64", True)
import gc
import sys
from time import time
import os

from energies import external_eg, exani_eg, hmag_eg, total_eg
from mapping import escript2arrays, check_external, check_anisotropy, check_hmag, update_pars, initial_m
from tools import write_stats, get_memory_usage, get_mu0
from store import pickle2jax
from escript_tools import readmesh_get_tags

import jax.numpy as np

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript chekc.py modelname")

    print('\n')
    print('MAP FINITE ELEMENT BACKEND (esys-escript) TO JAX \n')
    print('Memory before conversion to jax ', get_memory_usage(), "MB") 
    if os.path.exists(f'{name}.pkl'):
        print('read stored matrices')
        m, pars = pickle2jax(name)
        tags    = readmesh_get_tags(name)
        update_pars(name, pars)
        m = initial_m(tags.getDomain(),pars)
    else:
        m, pars, tags = escript2arrays(name,0)
    print('Memory after  conversion to jax ', get_memory_usage(), "MB") 
    gc.collect()
    print('Memory after  garbage collection', get_memory_usage(), "MB") 
    
    total_energy = 0

    total_energy += check_external(name,m,pars)  
    total_energy += check_anisotropy(name,m,pars)
    total_energy += check_hmag(name,m,pars)

    print('\n')
    print('COMPUTE TOTAL ENERGY')
    
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

    write_stats('timing and statistics',
                total_time, 0, function_calls,0)
                           
    print('\nenergy density of uniformly magnetized state')
    print('with jax backend (J/m^3)',energy/get_mu0())
    print('sum of terms     (J/m^3)',total_energy/get_mu0())


    
