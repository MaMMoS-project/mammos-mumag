from jax import jit
import jax.numpy as np

@jit
def normalize_vectors(m):
    vectors = m.reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / np.where(norms == 0, 1.0, norms)
    return normalized_vectors.reshape(-1)
    
@jit
def update_x(x,step,d):
   return x + step*d
   
@jit
def update_m(m,step,d):
   m1 =  m + step*d
   return normalize_vectors(m1)

@jit   
def dot_magnetizations(a, b):
    a_vectors = a.reshape((-1, 3))
    b_vectors = b.reshape((-1, 3))
    return np.sum(a_vectors * b_vectors, axis=1, keepdims=True)
