import sys

import esys.escript as e
from esys.weipa import saveVTK

from tools import read_params, dot
from magnetization import getM
from materials import Materials
from hmag import Hmag
from exani import ExAni
from external import External
from minimize import Minimize

class Loop:
  
  def __init__(self,params,materials):
    
    m, h, start, final, step, self._mstep, self._mfinal, min_params = params    
    self._external = External(start, final, step, h, materials.meas, materials.volume)
    exani = ExAni(materials.A, materials.K, materials.u, materials.volume)
    hmag_on, truncation, tol_u, tol_mxh, verbose = min_params
    if hmag_on==1:  
      hmag = Hmag(materials.Js, materials.volume, tol_u, verbose)
    else:
      hmag = None
    self._direction = h 
    self._meas = materials.meas 
    self._volume = materials.volume
    self._materials = materials
      
    self._minimize = Minimize(self._external,exani,hmag, min_params[1:])
  
  def compute_mh(self,m):
    return dot((self._direction[0]*m[0] + self._direction[1]*m[1] + self._direction[2]*m[2]),self._meas)/self._volume
  
  def solve(self,m,name):
    with open(name+'.dat','w') as f:
      i = 0
      mh_old = 9.99e9
      while self._external.next():
        m  = self._minimize.solve(m)
        mh = self.compute_mh(m)
        print("-dem->  ",self._external.value,mh,flush=True)
        if abs(mh - mh_old) >= self._mstep:
          mh_old = mh
          i = i+1
          saveVTK(name+f'.{i:04}',tags=self._materials.get_tags(),m=m)
          f.write(f'{i:04} {self._external.value} {mh}\n')
        else:
          f.write(f'{0:04} {self._external.value} {mh}\n')
        if mh < self._mfinal:
          break

  def getStatistics(self):
    return self._minimize.getStatistics() 




if __name__ == '__main__':
  try:
    name = sys.argv[1]        
  except IndexError:
    sys.exit("usage run-escript loop.py modelname")

  params = read_params(name)    
  materials = Materials(name)
  m = getM(e.wherePositive(materials.meas),params[0])
  
  loop = Loop(params,materials)
  loop.solve(m,name)
  out = loop.getStatistics()
  print(out)
