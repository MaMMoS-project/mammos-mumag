from esys.escript import normalize
from esys.escript.minimizer import (
    AbstractMinimizer,
    EvalutedPhi,
    CostFunction1DEvaluationFactory,
    LineSearchTerminationError,
    MinimizerIterationIncurableBreakDown,
)  # MinimizerMaxIterReached


class LBFGS(AbstractMinimizer):
    def __init__(self, F=None, m_tol=1e-4, grad_tol=1e-8, iterMax=300, logger=None):
        self.total_iter = 0
        super().__init__(
            F=F, m_tol=m_tol, grad_tol=grad_tol, iterMax=iterMax, logger=logger
        )

    def run(self, m):
        """
        :param m: initial guess
        :type m: m-type
        :return: solution
        :rtype: m-type
        """
        assert self._m_tol > 0.0
        assert self._grad_tol > 0.0
        assert self._iterMax > 1
        assert self._relAlphaMin > 0.0
        assert self._truncation > 0
        assert self._restart > 0
        # start the iteration:
        iterCount = 0
        iterCount_last_break_down = -1

        alpha = self._initialAlpha
        H_scale = None

        args_m = self.getCostFunction().getArgumentsAndCount(m)
        grad_Fm = self.getCostFunction().getGradientAndCount(m, *args_m)
        norm_m = self.getCostFunction().getNormAndCount(m)
        Fm = self.getCostFunction().getValueAndCount(m, *args_m)
        self._result = (m, Fm)

        Fm_old = Fm
        self.logger.info("Initialization completed.")

        self.doCallback(
            iterCount=0,
            m=m,
            dm=None,
            Fm=Fm,
            grad_Fm=grad_Fm,
            norm_m=norm_m,
            norm_gradFm=None,
            args_m=args_m,
            failed=False,
        )

        non_curable_break_down = False
        converged = False
        while (
            not converged and not non_curable_break_down and iterCount < self._iterMax
        ):
            k = 0
            break_down = False
            s_and_y = []
            self.__initializeHessian = True

            while (
                not converged
                and not break_down
                and k < self._restart
                and iterCount < self._iterMax
            ):
                self.logger.info("********** iteration %3d **********" % iterCount)
                self.logger.info("\tF(m) = %g" % Fm)
                # determine search direction
                p = -self._twoLoop(H_scale, grad_Fm, s_and_y, m, args_m)
                # Now we call the line search with F(m+alpha*p)
                # at this point we know that grad F(m) is not zero?
                phi = EvalutedPhi(
                    CostFunction1DEvaluationFactory(
                        m, p, costfunction=self.getCostFunction()
                    ),
                    alpha=0.0,
                    args=args_m,
                    valF=Fm,
                    gradF=grad_Fm,
                )
                if norm_m > 0:
                    alphaMin = (
                        self._relAlphaMin
                        * norm_m
                        / self.getCostFunction().getNormAndCount(p)
                    )
                else:
                    alphaMin = self._relAlphaMin
                self.getLineSearch().setOptions(alphaMin=alphaMin)
                alpha = max(alpha, alphaMin * 1.10)
                self.logger.debug(
                    "Starting line search with alphaMin, alpha  = %g, %g"
                    % (alphaMin, alpha)
                )
                alpha = 1.0
                try:
                    phi_new = self.getLineSearch().run(phi, alpha)
                    alpha = phi_new.alpha
                    Fm_new = phi_new.valF
                    grad_Fm_new = phi_new.gradF
                    args_m_new = phi_new.args
                except LineSearchTerminationError as e:
                    self.logger.debug(
                        "Line search failed: %s. BFGS is restarted." % str(e)
                    )
                    break_down = True
                    break
                # this function returns a scaling alpha for the search
                # direction as well as the cost function evaluation and
                # gradient for the new solution approximation x_new=x+alpha*p
                self.logger.debug("Search direction scaling found as alpha=%g" % alpha)
                # execute the step
                delta_m = alpha * p
                # m_new = m + delta_m
                m_new = normalize(m + delta_m)

                norm_m_new = self.getCostFunction().getNormAndCount(m_new)
                norm_dm = self.getCostFunction().getNormAndCount(delta_m)

                self._result = (m_new, Fm_new)
                converged = False

                """
                mtol_abs = norm_m_new * self._m_tol
                flag = norm_dm <= mtol_abs
                if flag:
                    self.logger.info("F(m) = %g" % Fm_new)
                    self.logger.info("Solution has converged: dm=%g, m*m_tol=%g" % (norm_dm, mtol_abs))
                    converged = True
                    break
                else:
                    self.logger.info("Solution checked: dx=%g, x*m_tol=%g" % (norm_dm, mtol_abs))
                # unfortunately there is more work to do!
                """

                if grad_Fm_new is None:
                    self.logger.debug("Calculating missing gradient.")
                    args_new = self.getCostFunction().getArgumentsAndCount(m_new)
                    grad_Fm_new = self.getCostFunction().getGradientAndCount(
                        m_new, *args_new
                    )

                """
                Ftol_abs = self._grad_tol * abs(max(abs(Fm), abs(Fm_new)))
                gradNorm1 = abs(self.getCostFunction().getDualProductAndCount(m_new, grad_Fm_new))/norm_m_new
                gradNorm2 = abs(self.getCostFunction().getDualProductAndCount(delta_m, grad_Fm_new))/norm_dm
                gradNorm=max(gradNorm1, gradNorm2)
                flag = gradNorm <= Ftol_abs
                if flag:
                    converged = True
                    self.logger.info("F(m) = %g" % Fm_new)
                    self.logger.info("grad Fm = %g, %g" % (gradNorm1, gradNorm2))
                    self.logger.info("Gradient has converged: grad F=%g, grad_tol=%g" % (gradNorm, Ftol_abs))
                    break
                else:
                    self.logger.info("grad Fm = %g, %g" % (gradNorm1, gradNorm2))
                    self.logger.info("Gradient checked: grad F=%g, grad_tol=%g" % (gradNorm, Ftol_abs))
                """

                mxh = self.getCostFunction().grad2mxh(grad_Fm_new)
                # print(k,Fm_new,mxh,alpha,self.getCostFunction().Arguments_calls)
                flag = mxh <= self._grad_tol
                if flag:
                    converged = True
                    self.logger.info(
                        f"energy, mxh = {Fm_new}, {mxh}, ({self._grad_tol}), converged"
                    )
                    break
                else:
                    self.logger.info(
                        f"energy, mxh = {Fm_new}, {mxh}, ({self._grad_tol})"
                    )

                delta_g = grad_Fm_new - grad_Fm
                rho = self.getCostFunction().getDualProductAndCount(delta_m, delta_g)
                if abs(rho) > 0:
                    s_and_y.append((delta_m, delta_g, rho))
                else:
                    self.logger.debug("Break down detected (<dm,dg>=0).")
                    break_down = True
                    break
                # move forward
                m = m_new
                grad_Fm = grad_Fm_new
                Fm = Fm_new
                args_m = args_m_new
                norm_m = norm_m_new
                k += 1
                iterCount += 1
                self.doCallback(
                    iterCount=iterCount,
                    m=m,
                    dm=delta_m,
                    Fm=Fm,
                    grad_Fm=grad_Fm,
                    # norm_m=norm_m, norm_gradFm=gradNorm, args_m=args_m, failed=break_down)
                    norm_m=norm_m,
                    norm_gradFm=mxh,
                    args_m=args_m,
                    failed=break_down,
                )

                # delete oldest vector pair
                if k > self._truncation:
                    s_and_y.pop(0)

                if not break_down:
                    # an estimation of the inverse Hessian if it would be a scalar P=H_scale:
                    # the idea is that
                    #
                    #      H*dm=dg
                    #
                    #  if there is a inverse Hessian approximation provided we use
                    #
                    #     H_scale*|dm|^2 = <dm, dg>=rho ->   H_scale =rho/|dm|^2
                    H_scale = rho / norm_dm**2
                    self.logger.info("Scale of Hessian = %g." % H_scale)
            # case handling for inner iteration:
            if break_down:
                if not iterCount > iterCount_last_break_down:
                    non_curable_break_down = True
                    self.logger.warning(
                        ">>>>> Incurable break down detected in step %d." % iterCount
                    )
                else:
                    iterCount_last_break_down = iterCount
                    self.logger.debug(
                        "Break down detected in step %d. Iteration is restarted."
                        % iterCount
                    )
            if not k < self._restart:
                self.logger.debug("Iteration is restarted after %d steps." % iterCount)

        # case handling for inner iteration:
        if iterCount >= self._iterMax:
            self.logger.warning(
                ">>>>>>>>>> Maximum number of iterations reached! <<<<<<<<<<"
            )
            raise MinimizerMaxIterReached("Gave up after %d steps." % iterCount)
        elif non_curable_break_down:
            self.logger.warning(">>>>>>>>>> Incurable breakdown! <<<<<<<<<<")
            raise MinimizerIterationIncurableBreakDown(
                "Gave up after %d steps." % iterCount
            )
        self.logger.info("Success after %d iterations!" % iterCount)
        self.total_iter += iterCount
        return self._result

    def _twoLoop(self, H_scale, grad_Fm, s_and_y, m, args_m):
        """
        Helper for the L-BFGS method.
        See 'Numerical Optimization' by J. Nocedal for an explanation.
        """
        q = grad_Fm
        alpha = []
        for s, y, rho in reversed(s_and_y):
            a = self.getCostFunction().getDualProductAndCount(s, q) / rho
            alpha.append(a)
            q = q - a * y

        if self.__initializeHessian:
            self.logger.debug("Hessian is expected to be updated.")
        p = self.getCostFunction().getInverseHessianApproximationAndCount(
            q, m, initializeHessian=self.__initializeHessian, *args_m
        )
        self.__initializeHessian = False
        norm_p = self.getCostFunction().getNormAndCount(p)
        if not norm_p > 0:
            raise MinimizerException("Approximate Hessian inverse returns zero.")
        # this is if one wants
        if H_scale is not None and self._scaleSearchDirection:
            p_dot_q = self.getCostFunction().getDualProductAndCount(p, q)
            if abs(p_dot_q) > 0:
                #  we  would like to see that  h * norm(p)^2 = < H p, p> = < q, p> with h=norm(H)=H_scale
                # so we rescale p-> a*p ; h * a^2 * norm(p)^2 = a * < q, p> -> a = <q,p>/h/norm(p)**2
                a = abs(p_dot_q) / H_scale / norm_p**2
                p *= a
                self.logger.debug(
                    "Search direction scaled : p=%g (scale factor = %g)"
                    % (norm_p * a, a)
                )
        for s, y, rho in s_and_y:
            beta = self.getCostFunction().getDualProductAndCount(p, y) / rho
            a = alpha.pop()
            p += s * (a - beta)
        return p
