import esys.escript as e
import math

def getVortex(mask):
  domain = mask.getDomain()
  m = e.Vector(0,e.Solution(domain))
  x = domain.getX()
  r = e.sqrt(x[1]*x[1]+x[2]*x[2])
  m[0] = e.whereNegative(r-2.0)
  m[2] = e.whereNonNegative(r-2.0)*(e.whereNegative(x[1])-e.whereNonNegative(x[1]))
  return m 

def getM(mask,v):
  if sum(v)==0.0:
    m = mask*getVortex(mask)
  else:    
    m = mask*e.Vector(v,e.Solution(mask.getDomain()))
  return e.normalize(m)

def xM(mask,scale):
  domain = mask.getDomain()
  m = e.Vector(0,e.Solution(domain))
  x = domain.getX()
  k = 0.5*math.pi/scale
  kr = k*(x[0] + x[1])
  m[0] = e.sin(kr)
  m[1] = e.cos(kr)
  m[2] = 0
  return mask*m,k

'''
exchange energy:
  mx  =   sin(kr)
  my  =   cos(kr)
  dmx              dmy
  --- = k*cos(kr), --- = - k*sin(kr)
  dx               dx
  dmx              dmy
  --- = k*cos(kr), --- = - k*sin(kr)
  dy               dy
  mx,i mx,i = k*k*cos2 + k*k*cos2 = 2 k*k *cos2
  my,i my,i = k*k*sin2 + k*k*sin2 = 2 k*k *sin2
  eex = 2 A k^2
'''
