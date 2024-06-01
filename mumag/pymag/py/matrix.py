from esys.escript.linearPDEs import LinearSinglePDE, LinearPDE
import esys.escript as e

def grad_matrix(Js,volume,v):
  pde = LinearSinglePDE(Js.getDomain())
  pde.setValue(C=(Js/volume)*v)
  return pde.getOperator()

def gx(Js,volume):
  return grad_matrix(Js,volume,[1,0,0])

def gy(Js,volume):
  return grad_matrix(Js,volume,[0,1,0])

def gz(Js,volume):
  return grad_matrix(Js,volume,[0,0,1])
  
def exani_matrix(A,K,u,volume):
  if A:
    domain = A.getDomain()
  else:
    domain = K.getDomain()
  pde = LinearPDE(domain)
  if A:
    pde.setValue(A=(2.0*A/volume)*e.identityTensor4(domain))
  if K:
    pde.setValue(D=(-2.0*K/volume)*e.outer(u,u))
  return pde.getOperator()
    
