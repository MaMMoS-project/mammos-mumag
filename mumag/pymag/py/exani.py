import sys
import math
from time import time

import esys.escript as e

from materials import Materials
from magnetization import xM, getM
from matrix import exani_matrix
from tools import dot, read_A, readAnisotropyEnergy
  
class ExAni:
  def __init__(self, A, K, u, volume):
    self._matrix = exani_matrix(A, K, u, volume)
    self.cum_time = 0.
  
  def solve_g(self, m):
    T0 = time() 
    g = self._matrix.of(m)
    self.cum_time += time()-T0
    return g
    
  def solve_e(self, m):  
    return 0.5*dot(m,self.solve_g(m))
    
  def saveMM(self,fn):
    self._matrix.saveMM(fn)
    
if __name__ == '__main__':
  try:
    name = sys.argv[1]        
  except IndexError:
    sys.exit("usage run-escript hmag.py modelname")
    
  materials = Materials(name)

  exani = ExAni(materials.A,materials.K,materials.u,materials.volume)

  scale = materials.volume**0.3333333333333333
  size  = materials.size
  m, k = xM(e.wherePositive(materials.meas),scale)
  
  A = read_A(name)
  mu0 = 4.0e-7 * math.pi

  print("vortex on a, b plane")
  print("energy from gradient",exani.solve_e(m))
  print("       analytic     ",2*k*k*mu0*A/(size*size))

  uniform_m = [0.,0.,1.]
  m = getM(e.wherePositive(materials.meas),uniform_m)
  eani = readAnisotropyEnergy(name,uniform_m)

  print("uniform m",uniform_m)
  print("energy from gradient",exani.solve_e(m))
  print("       analytic     ",eani)

