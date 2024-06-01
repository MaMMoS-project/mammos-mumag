import esys.escript as e
from esys.escript.minimizer import CostFunction
from lbfgs import LBFGSM

from tools import dot, Linf_norm, L2_norm, get_logger

import logging
from time import time

class MumagFunc(CostFunction):
    def __init__(self, external,exani,hmag):
        super().__init__()
        self._external = external
        self._exani = exani
        self._hmag = hmag
        self.dot_time = 0.
        self.norm_time = 0.
        self.energies_time = 0.

    def getDualProduct(self,  m, g):
        T0 = time()
        d = dot(g, m)
        self.dot_time += time()-T0
        return d
        
    def getArguments(self, m):
        T0 = time()
        mn = e.normalize(m)
        if self._hmag:
          a = (mn, self._external.solve_g(mn), self._exani.solve_g(mn), self._hmag.solve_g(mn))
        else:
          a = (mn, self._external.solve_g(mn), self._exani.solve_g(mn), None)
        self.energies_time += time()-T0
        return a
        
    def getValue(self, m, *args):
        T0 = time()
        mn, g_ext, g_exani, g_hmag = args
        if g_hmag:
          value = dot(mn,g_ext) + 0.5*dot(mn,g_exani) + 0.5*dot(mn,g_hmag)
        else:
          value = dot(mn,g_ext) + 0.5*dot(mn,g_exani)
        self.energies_time += time()-T0
        return value

    def getGradient(self, m, *args):
        T0 = time()
        mn, g_ext, g_exani, g_hmag = args
        if g_hmag:
          g = e.cross(mn,e.cross(g_ext+g_exani+g_hmag,mn))
        else:
          g = e.cross(mn,e.cross(g_ext+g_exani,mn))
        self.energies_time += time()-T0
        return g
   
    def getNorm(self, m):
        T0 = time()
        n = Linf_norm(m)
        self.norm_time += time()-T0
        return n
        
    def getL2Norm(self, m):
        T0 = time()
        n = L2_norm(m)
        self.norm_time += time()-T0
        return n

    def grad2mxh(self, g):
        T0 = time()
        n = Linf_norm(g/(e.whereZero(self._external._meas)+self._external._meas))
        self.norm_time += time()-T0
        return n
        
    def getL2NormAndCount(self, m):
        """
        returns the norm of ``m``.

        When calling this method the calling statistics is updated.

        :type m: m-type
        :rtype: ``float``
        """
        self.L2Norm_calls += 1
        return self.getL2Norm(m)
        
    def resetStatistics(self):
        """
        resets all counters
        """
        self.DualProduct_calls = 0
        self.Value_calls = 0
        self.Gradient_calls = 0
        self.Arguments_calls = 0
        self.InverseHessianApproximation_calls = 0
        self.Norm_calls = 0    
        self.L2Norm_calls = 0
        
    def getStatistics(self,minimizer_iter,minimizer_time):
        """
        return the call statistics as a string:
        """
        out="\n"
        out+=f"lbfgs iterations                    : {minimizer_iter}\n"
        out+="number of inner product evaluations : %d\n" % self.DualProduct_calls
        out+="number of norm evaluations          : %d\n" % self.Norm_calls
        out+="number of L2 norm evaluations       : %d\n" % self.L2Norm_calls
        out+="number of argument evaluations      : %d\n" % self.Arguments_calls
        out+="number of cost function evaluations : %d\n" % self.Value_calls
        out+="number of gradient evaluations      : %d\n" % self.Gradient_calls
        out+=f"minimizer time                      : {minimizer_time}\n"
        out+=f"  inner products                    : {self.dot_time}\n"
        out+=f"  norms                             : {self.norm_time}\n"
        out+=f"  function and gradient             : {self.energies_time}\n"
        solver_iter, solver_time = self._hmag.getDiagnostics()
        out+=f"    zeeman                          : {self._external.cum_time}\n"
        out+=f"    exchange+anistorpy              : {self._exani.cum_time}\n"
        out+=f"    magnetostatic                   : {self._hmag.cum_time}\n"
        out+=f"      right hand side               : {self._hmag.rhs_time}\n"
        out+=f"      potential                     : {self._hmag.u_time}\n"
        out+=f"      gradient                      : {self._hmag.grad_time}\n"
        out+=f"    linear system iterations        : {solver_iter}"   
        return out
 
class Minimize:
  def __init__(self,external,exani,hmag,min_params):
    self.cum_time = 0.
    trunction, m_tol, grad_tol, verbose = min_params
    self.F = MumagFunc(external,exani,hmag)    
    self._solver = LBFGSM(self.F,logger=get_logger('min',verbose))
    self._solver.setOptions(truncation=trunction,m_tol=m_tol,grad_tol=grad_tol,scaleSearchDirection=False)
    
  def solve(self,m):
    T0 = time()
    self._solver.run(m)
    self.cum_time += time()-T0
    return self._solver.getResult()

  def getStatistics(self):
    return self.F.getStatistics(self._solver.total_iter,self.cum_time) 
