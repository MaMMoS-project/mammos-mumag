import esys.escript as e
from esys.escript.linearPDEs import LinearSinglePDE

import configparser
import numpy as np
import math
import logging


def normalize(v):
    a = v[0]
    b = v[1]
    c = v[2]
    s = math.sqrt(a * a + b * b + c * c)
    if s == 0.0:
        return [a, b, c]
    else:
        return [a / s, b / s, c / s]


def dot(a, b):
    np_a = e.convertToNumpy(a)
    np_b = e.convertToNumpy(b)
    return np.dot(np_b.flatten(), np_a.flatten())


def Linf_norm(a):
    np_a = e.convertToNumpy(a)
    return np.linalg.norm(np_a, np.inf)


def L2_norm(a):
    np_a = e.convertToNumpy(a)
    return np.linalg.norm(np_a)


def get_meas(Js):
    domain = Js.getDomain()
    pde = LinearSinglePDE(domain)
    pde.setValue(Y=Js)
    return pde.getRightHandSide()


def read_params(params_fname):
    config = configparser.ConfigParser(
        {
            "state": "mxyz",
            "mstep": 0.01,
            "mfinal": -0.8,
            "hmag_on": 1,
            "truncation": 5,
            "tol_fun": 1e-10,
            "tol_hmag_factor": 1.0,
            "precond_iter": 10,
            "iter_max": 1000,
        }
    )
    config.read(params_fname)
    intial_state = config["initial state"]
    field = config["field"]
    minimizer = config["minimizer"]
    m = normalize(
        [
            float(intial_state["mx"]),
            float(intial_state["my"]),
            float(intial_state["mz"]),
        ]
    )
    state = intial_state["state"]
    h = normalize([float(field["hx"]), float(field["hy"]), float(field["hz"])])
    hstart, hfinal, hstep = (
        float(field["hstart"]),
        float(field["hfinal"]),
        float(field["hstep"]),
    )
    mstep, mfinal = float(field["mstep"]), float(field["mfinal"])
    hmag_on = int(minimizer["hmag_on"])
    truncation = int(minimizer["truncation"])
    tol_fun = float(minimizer["tol_fun"])
    tol_hmag_factor = float(minimizer["tol_hmag_factor"])
    precond_iter = int(minimizer["precond_iter"])
    iter_max = int(minimizer["iter_max"])
    tol_u = tol_fun * tol_hmag_factor
    tol_mxh = tol_fun**0.3333333333333333333333333
    verbose = int(minimizer["verbose"])
    print(f"tolerances: optimality tolerance {tol_fun}   hmag {tol_u}   mxh {tol_mxh}")
    return (
        m,
        state,
        h,
        hstart,
        hfinal,
        hstep,
        mstep,
        mfinal,
        (hmag_on, truncation, tol_u, tol_mxh, precond_iter, iter_max, verbose),
    )


def get_logger(name, verbose):
    mylogger = logging.getLogger("min")
    if verbose == 0:
        mylogger.setLevel(logging.ERROR)
    if verbose == 1:
        mylogger.setLevel(logging.WARNING)
    if verbose == 2:
        mylogger.setLevel(logging.INFO)
    if verbose >= 3:
        mylogger.setLevel(logging.DEBUG)
    return mylogger

    """
        setOptions for LBFGS.  use       solver.setOptions( key = value)

        :key m_tol: relative tolerance for solution `m` for termination of iteration
        :type m_tol: `float`
        :default m_tol: 1e-4
        :key grad_tol: tolerance for gradient relative to initial costfunction value
                       for termination of iteration
        :type grad_tol: `float`
        :default grad_tol: 1e-8
        :key truncation: sets the number of previous LBFGS iterations to keep
        :type truncation : `int`
        :default truncation: 30
        :key restart: restart after this many iteration steps.
        :type restart: `int`
        :default restart: 60
        :key iterMax: maximium number of iterations.
        :type iterMax: `int`
        :default iterMax: 300
        :key relAlphaMin: minimal step size relative to serach direction.
                          The value should be chosen such that
                          At any iteration step `F(m + alpha * p)` is just
                          discriminable from `F(m)` for any
                          `alpha > relAlphaMin * |m|/|p|'.
        :type relAlphaMin: ``float``
        :default relAlphaMin: 1e-8
        :key initialAlpha: initial step size alpha in line serach. Typically alpha=1
                           is a good initial value but a larger or smaller initial
                           value may help to get the iteration started when only an
                           approximation of the Hessian is available.
        :type initialAlpha: ``float``
        :default initialAlpha: 1.
        :key scaleSearchDirection: if set the search direction is rescaled using an
                                   estimation of the norm of the Hessian
        :type scaleSearchDirection: ``bool``
        :default scaleSearchDirection: True

            Example of usage::
              cf=DerivedCostFunction()
              solver=MinimizerLBFGS(J=cf, m_tol = 1e-5, grad_tol = 1e-5, iterMax=300)
              solver.setOptions(truncation=20)
              solver.getLineSearch().setOptions(zoom_relChangeMin =1e-7)
              solver.run(initial_m)
              result=solver.getResult()
  """

    """
        set options for the line search.


        :key alphaMin: minimum search length
        :type alphaMin: float
        :default alphaMin: 1e-20
        :key alphaMax: maximum search length
        :type alphaMax: float
        :default alphaMax: 5000
        :key overStepFactor : factor to increase step size in line search
        :type overStepFactor: float
        :default overStepFactor: 2.
        :key iterMax: maximum number of line search iterations
        :type iterMax: int
        :default iterMax: 25
        :key c1: sufficient decrease condition factor c1
        :type c1: float
        :default c1: 1e-4
        :key c2: curvature condition factor c2
        :type c2: float
        :default c2: 0.9
        :key inter_order: order of the interpolation used for line search
        :type inter_order: 1,2,3
        :default inter_order: 3
        :key inter_iterMax: maximum number of iteration steps to when minimizing
                            interploted cost function
        :type inter_iterMax: int
        :default inter_iterMax: 100
        :key inter_tol: tolerance to when minimizing interploted cost function
        :type inter_tol: float
        :default inter_tol: 1.
        :key zoom_iterMax: maximum number of zoom iterations
        :type zoom_iterMax: int
        :default zoom_iterMax: 20

        :key phiEpsilon : tolerance for `greater than` check of cost function values
        :type phiEpsilon: float
        :default phiEpsilon: ``np.sqrt(EPSILON)``

        :key  alphaOffset : minimal relative distance of new alpha from boundaries
        :type alphaOffset : float
        :default alphaOffset : 1e-4

        :key alphaWidthMin : minimal relative distance of new alpha from boundaries
        :type alphaWidthMin: float
        :default alphaWidthMin: ``np.sqrt(EPSILON)``
        :key zoom_reductionMin: minimal reduction search interval length between zoom
                                steps
        :type zoom_reductionMin : float
        :default zoom_reductionMin :0.66
  """


"""
Can I access local value in Data objects from python?

Yes, but I don't recommend to do this!

This little script shows you how to do this:

     in=Data(..)
    out=Data(..)
     for i in xrange(in.getNumberOfDataPoints())
           in_loc=in.getValueOfDataPoint(i)
          assert isinstance(in_loc, numarray.NumArray)
          out_loc=< some operations on in_loc>
          out.setValueOfDataPoint(i,out_loc)

Notice that in_loc and out_loc are numarray objects with same ranks and shape
like in and out. in and out both need to be defined on the same FunctionSpace.
"""

"""
How do I use tagged Data ?

A Here comes an example of tagged scalar data for a Domain mydom:

    s=Scalar(0.,Function(mydom))
    s.setTaggedValue(1000,1.)
    s.setTaggedValue(2000,2.)

In this case the Data object s uses 1. for all samples which have been tagged with the
value 1000 and the value 2. for samples tagged with 2000. All other samples use the
dafault value 0. set tn the Scalar() call. In case of finley, the FunctionSpace
Function is represented by elements so the tag refer to the tags assigned to the
elements, typically during mesh generation.

There is a way to modify the tags after the mesh as been generated: If you want to set
tag 1000 for element left from x0=0.5 and tag 2000 for element right from x0=0.5 you
can use:

x0=Function(mydom).getX()[0]
Function(mydom).setTags(1000,whereNegative(x0-0.5))
Function(mydom).setTags(2000,whereNegative(0.5-x0))

Note that although setTags is called for different instances of Function(mydom) is
still changes the element tag as it effects the underlying mydom. It also important
to point out that a tag sticks to the element even if the element coordinates are
altered by a mydom.setX() call and is considered when the mesh is written to a file.
"""

"""
How can I integrate a function over the boundary or parts of the boundary?

To integrate a function f over the boundary use

   integrate(f,where=FunctionOnBoundary(f.getDomain()))

Be aware that f must be defined on the boundary or must be interpolatable to the
boundary (WARNING: In the case of finley, the boundary is defined by the face elements)

If you want to integrate over parts of the boundary you can use masks: For instance to
integrate over the portion of the boundary where x_0==0 use

    integrate(f*whereZero(FunctionOnBoundary(f.getDomain()).getX()[0])))
"""

"""
When performing a binary operation such as a+b where a is a numarray.NumArray object
and b is an escript.Data object I get the error message

TypeError: UFunc arguments must be numarray, scalars or numeric sequences

What does it mean?

Unfortunatley there is a bug in numarray as binary operations in numarray don't handle
non-numarray arguments properly. This stops python from calling an appropriate escript
function to handle the problem namely by calling the corresponding add opertaor of the
non-numarray argument. The only solution is to avoid any expression starting with a
numarray object. Alternatively you can use the add, mult, div, power functions provided
by escript. (Remark: python list objects handle type mismatches properly. You can write
[1.,1.]+Vector(...) but not numarray.ones((2,))+Vector(...). Nevertheless,
add([1.,1.],Vector(...)) and add(numarray.ones((2,)),Vector(...)) will do the job
but don't look nice).
"""
