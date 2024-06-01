import sys

from esys.escript import normalize
from esys.escript.minimizer import AbstractMinimizer, EvalutedPhi, CostFunction1DEvaluationFactory, LineSearchTerminationError, MinimizerIterationIncurableBreakDown

class LBFGSM(AbstractMinimizer):
    def __init__(self,F=None, m_tol=1e-4, grad_tol=1e-8, iterMax=300, logger=None):
        self.total_iter = 0
        super().__init__(F=F,m_tol=m_tol,grad_tol=grad_tol,iterMax=iterMax,logger=logger)


    def run(self, m):

        assert self._grad_tol > 0.
        assert self._iterMax > 1
        assert self._relAlphaMin > 0.
        assert self._truncation > 0
        assert self._restart > 0
        
        # Florian Mannel, A globalization of L-BFGS for nonconvex unconstrained optimization
        c0 = 1.0e-4
        c1 = 1.0
        c2 = 1.0 / (2 * self._truncation + 3)
        
        # start the iteration:
        iterCount = 0
        iterCount_last_break_down = -1

        alpha = self._initialAlpha

        self._result = m
        args_m = self.getCostFunction().getArgumentsAndCount(m)
        grad_Fm = self.getCostFunction().getGradientAndCount(m, *args_m)
        norm_m = self.getCostFunction().getNormAndCount(m)
        Fm = self.getCostFunction().getValueAndCount(m, *args_m)
        Fm_old = Fm
        self.logger.info("Initialization completed.")

        mxh = self.getCostFunction().grad2mxh(grad_Fm)
        flag = mxh <= self._grad_tol
        if flag:
            converged = True
            self.logger.info(f"energy, mxh = {Fm}, {mxh}, ({self._grad_tol}), converged")
            self._result = m
        else:
            self.logger.info(f"energy, mxh = {Fm}, {mxh}, ({self._grad_tol})")


        non_curable_break_down = False
        converged = False
        while not converged and not non_curable_break_down and iterCount < self._iterMax:
            k = 0
            gam = 1.0
            swv = 0
            om = 0.
            break_down = False
            s_and_y = []

            while not converged and not break_down and k < self._restart and iterCount < self._iterMax:
                self.logger.info("********** iteration %3d **********" % iterCount)
                self.logger.info("\tF(m) = %g" % Fm)
                
                # determine search direction
                p, usr = self._twoLoop(grad_Fm, s_and_y, m, gam, om)
                if k > 0:
                  self.logger.info(f"swv = {swv}, fraction of secant pairs used {usr/len(s_and_y)}.")                  
                phi=EvalutedPhi(CostFunction1DEvaluationFactory(m, p, costfunction=self.getCostFunction()),
                                alpha=0., args=args_m, valF=Fm, gradF=grad_Fm)
                # linesearch
                # if norm_m > 0:
                #    alphaMin = self._relAlphaMin * norm_m/self.getCostFunction().getNormAndCount(p)
                #else:
                #    alphaMin = self._relAlphaMin
                #self.getLineSearch().setOptions(alphaMin=alphaMin)
                #alpha = max(alpha, alphaMin*1.10)
                #self.logger.debug("Starting line search with alphaMin, alpha  = %g, %g" % (alphaMin, alpha))
                alpha = 1.0
                try:
                    phi_new = self.getLineSearch().run(phi, alpha)
                    alpha = phi_new.alpha
                    Fm_new = phi_new.valF
                    grad_Fm_new = phi_new.gradF
                    args_m_new = phi_new.args
                except LineSearchTerminationError as e:
                    self.logger.debug("Line search failed: %s. BFGS is restarted." % str(e))
                    break_down = True
                    break

                # this function returns a scaling alpha for the search
                # direction as well as the cost function evaluation and
                # gradient for the new solution approximation x_new=x+alpha*p
                self.logger.debug("Search direction scaling found as alpha=%g" % alpha)
                # execute the step
                delta_m = alpha * p
                m_new = normalize(m + delta_m)
                                
                norm_m_new = self.getCostFunction().getNormAndCount(m_new)
                norm_dm = self.getCostFunction().getNormAndCount(delta_m)

                self._result = m_new
                converged = False
                
                if grad_Fm_new is None:
                    self.logger.debug("Calculating missing gradient.")
                    args_new = self.getCostFunction().getArgumentsAndCount(m_new)
                    grad_Fm_new = self.getCostFunction().getGradientAndCount(m_new, *args_new)

                mxh = self.getCostFunction().grad2mxh(grad_Fm_new)
                flag = mxh <= self._grad_tol
                if flag:
                    converged = True
                    self.logger.info(f"energy, mxh = {Fm_new}, {mxh}, ({self._grad_tol}), converged")
                    break
                else:
                    self.logger.info(f"energy, mxh = {Fm_new}, {mxh}, ({self._grad_tol})")

                delta_g = grad_Fm_new - grad_Fm
                rho = self.getCostFunction().getDualProductAndCount(delta_m, delta_g)
                
                om = min([c0, c1 * self.getCostFunction().getL2NormAndCount(grad_Fm_new) ** c2])   # Mannel: Algorithm LBFGSM line 4, matlab code line 184
                nms2sc = self.getCostFunction().getDualProductAndCount(delta_m, delta_m)           # Mannel: matlab code line 186
                nmy2sc = self.getCostFunction().getDualProductAndCount(delta_g, delta_g)
                if rho > 0:
                  swv = 1
                  gam = rho / nmy2sc  # Classical choice of gamma_k
                  s_and_y.append((delta_m, delta_g, rho, nms2sc, nmy2sc))   # Mannel: matlab code lines 200 to 204
                elif nmy2sc > 0:
                  swv = 2
                  gam = np.sqrt(nms2sc / nmy2sc)  # If classical choice impossible, use this one
                else:
                  swv = 3
                  gam = 1.                       # If alternative choice of gamma_k is also impossible, use gamma_k = 1
                gam = max(om, min(gam, 1 / om))  # To ensure that gamma_k\in[\omega_k,1/\omega_k], Mannel: matlab code line 198
                
                # move forward
                m = m_new
                grad_Fm = grad_Fm_new
                Fm = Fm_new
                args_m = args_m_new
                norm_m = norm_m_new
                k += 1
                iterCount += 1

                # delete oldest vector pair
                if k > self._truncation:
                    s_and_y.pop(0)

            # case handling for inner iteration:
            if break_down:
                if not iterCount > iterCount_last_break_down:
                    non_curable_break_down = True
                    self.logger.warning(">>>>> Incurable break down detected in step %d." % iterCount)
                else:
                    iterCount_last_break_down = iterCount
                    self.logger.debug("Break down detected in step %d. Iteration is restarted." % iterCount)
            if not k < self._restart:
                self.logger.debug("Iteration is restarted after %d steps." % iterCount)

        # case handling for inner iteration:
        if iterCount >= self._iterMax:
            self.logger.warning(">>>>>>>>>> Maximum number of iterations reached! <<<<<<<<<<")
            raise MinimizerMaxIterReached("Gave up after %d steps." % iterCount)
        elif non_curable_break_down:
            self.logger.warning(">>>>>>>>>> Incurable breakdown! <<<<<<<<<<")
            raise MinimizerIterationIncurableBreakDown("Gave up after %d steps." % iterCount)
        self.logger.info("Success after %d iterations!" % iterCount)
        
        self.total_iter += iterCount
        
        return self._result

    def _twoLoop(self, grad_Fm, s_and_y, m, gam, om):
        """
        Helper for the L-BFGS method.
        See 'Numerical Optimization' by J. Nocedal for an explanation.
        """
        q = grad_Fm
        alpha = []
        for s, y, rho, nms2, nmy2 in reversed(s_and_y):
          if rho >= om * max(nms2, nmy2): # Mannel: matlab code line 220              
            a = self.getCostFunction().getDualProductAndCount(s, q) / rho
            alpha.append(a)
            q = q - a * y

        p = gam*q             # Mannel: Algorithm LBFGSM line 4 line 6
                
        usr = 0        
        for s, y, rho, nms2, nmy2  in s_and_y:
          if rho >= om * max(nms2, nmy2): # Mannel: matlab code line 220
            beta = self.getCostFunction().getDualProductAndCount(p, y) / rho
            a = alpha.pop()
            p += s * (a - beta)
            usr += 1
                
        return -p, usr
