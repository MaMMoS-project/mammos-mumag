import sys
import pickle
import numpy
from mapping import escript2arrays, pars2jax
import jax.numpy as np

# store parameters in a pickle file
def to_pickle(name,m,pars):
    with open(f"{name}.pkl", "wb") as f:
        pickle.dump(m, f)
        pickle.dump(pars, f)

def from_pickle(name):
    with open(f"{name}.pkl", "rb") as f:
        m    = pickle.load(f)
        pars = pickle.load(f)
    return m, pars
        
def pickle2jax(name):
    m, pars = from_pickle(name)
    m = np.array(m)
    pars2jax(pars)
    return m, pars

if __name__ == "__main__":
    try:
        name = sys.argv[1]
    except IndexError:
        sys.exit("usage run-escript store.py modelname")

    m, pars, _ = escript2arrays(name,0,'numpy')
    to_pickle(name, m, pars)
    m, pars = pickle2jax(name)

    
