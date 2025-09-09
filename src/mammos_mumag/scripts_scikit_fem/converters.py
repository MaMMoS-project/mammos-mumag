from jax.experimental import sparse
import jax.numpy as np
import scipy
import tempfile
import numpy

import esys.escript as e

# escript operator to scikit-sparse matrix
def operator2csr(operator):
  with tempfile.NamedTemporaryFile(suffix=".mtx", delete=True) as temp_file:
    filename = temp_file.name  # Get the temporary filename
    operator.saveMM(filename)  # Save matrix to temporary file
    return scipy.io.mmread(filename).tocsr()  

# scipy csr to jax BCOO
def csr2bcoo(A):
    return sparse.BCOO.from_scipy_sparse(A).sort_indices()
    
# escript operator to matrix
def operator2matrix(operator,fm='bcoo',diag=True):
    A = operator2csr(operator)
    if diag:
      D = np.array(A.diagonal())
    else:
      D = None        
    if fm.lower()=='bcoo':
      A = csr2bcoo(A) 
    return A, D

# escript array to numpy
def escript2numpy(v):
    return numpy.array(e.convertToNumpy(v).T.flatten())

# escript array to jax
def escript2jax(v):
    return np.array(e.convertToNumpy(v).T.flatten())

# escript array to numpy
def escript2numpy(v):
    return e.convertToNumpy(v).T.flatten()

# jax.np array of shape (3N,) to escript Vector
def toEscriptVector(x,domain):
    y = numpy.array(x.reshape(-1,3))  
    m = e.Vector(0.,e.Solution(domain))
    for i,mm in enumerate(y):
       m.setValueOfDataPoint(i,mm)
    return m

# jax.np array of shape (N,) to escript Scalar
def toEscriptScalar(x,domain):
    y = numpy.array(x)  
    u = e.Scalar(0.,e.Solution(domain))
    for i,uu in enumerate(y):
       u.setValueOfDataPoint(i,uu)
    return u
