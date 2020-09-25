import warnings
import torch
from torch import tensor, Tensor

# from scipy
def wrap_function(function, args):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(*wrapper_args):
        ncalls[0] += 1
        return function(*(wrapper_args + args))

    return ncalls, function_wrapper

# Not working yet
def minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=tensor(1e-4), fatol=tensor(1e-4), adaptive=False,
                         **unknown_options):
    """
    A port of scipy's nelder-mead optimizer
    Minimization of scalar function of one or more variables using the
    Nelder-Mead algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter, maxfev : int
        Maximum allowed number of iterations and function evaluations.
        Will default to ``N*200``, where ``N`` is the number of
        variables, if neither `maxiter` or `maxfev` is set. If both
        `maxiter` and `maxfev` are set, minimization will stop at the
        first reached.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    initial_simplex : array_like of shape (N + 1, N)
        Initial simplex. If given, overrides `x0`.
        ``initial_simplex[j,:]`` should contain the coordinates of
        the jth vertex of the ``N+1`` vertices in the simplex, where
        ``N`` is the dimension.
    xatol : float, optional
        Absolute error in xopt between iterations that is acceptable for
        convergence.
    fatol : number, optional
        Absolute error in func(xopt) between iterations that is acceptable for
        convergence.
    adaptive : bool, optional
        Adapt algorithm parameters to dimensionality of problem. Useful for
        high-dimensional minimization [1]_.
    References
    ----------
    .. [1] Gao, F. and Han, L.
       Implementing the Nelder-Mead simplex algorithm with adaptive
       parameters. 2012. Computational Optimization and Applications.
       51:1, pp. 259-277
    """
    if 'ftol' in unknown_options:
        print("ftol is deprecated for Nelder-Mead,"
                      " use fatol instead. If you specified both, only"
                      " fatol is used.",
                      DeprecationWarning)
        if (torch.isclose(fatol, tensor(1e-4)) and
                not torch.isclose(unknown_options['ftol'], tensor(1e-4))):
            # only ftol was probably specified, use it.
            fatol = unknown_options['ftol']
        unknown_options.pop('ftol')
    if 'xtol' in unknown_options:
        print("xtol is deprecated for Nelder-Mead,"
                      " use xatol instead. If you specified both, only"
                      " xatol is used.",
                      DeprecationWarning)
        if (torch.isclose(xatol, tensor(1e-4)) and
                not torch.isclose(unknown_options['xtol'], tensor(1e-4))):
            # only xtol was probably specified, use it.
            xatol = unknown_options['xtol']
        unknown_options.pop('xtol')
    
    #TODO: restor
    # _check_unknown_options(unknown_options)
    if unknown_options:
        warnings.warn(f"Unknown options: {', '.join(map(str, unknown_options.keys()))}")
    maxfun = maxfev
    retall = return_all

    # TODO: need to convert arguments to tuple? currently passed as dict
    fcalls, func = wrap_function(func, args)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    x0 = tensor(x0, dtype=torch.float32).flatten()

    if initial_simplex is None:
        N = len(x0)

        sim = torch.zeros((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = tensor(x0) #originally np.array(x0, copy=True), but x = [[1,2,3],[4,5,6]]; y = tensor(x); z = tensor(x); y[0,1] = 15; y[0,1] != x[0,1]
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = tensor(initial_simplex, dtype=torch.float32) #originally np.asfarray(initial_simplex).copy()
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]
    
    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = tensor(N * 200)
        maxfun = tensor(N * 200)
    elif maxiter is None:
        maxfun = tensor(maxfun)
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if torch.isinf(maxfun):
            maxiter = tensor(N * 200)
        else:
            maxiter = tensor(float('inf'))
    elif maxfun is None:
        maxiter = tensor(maxiter)
        if torch.isinf(maxiter):
            maxfun = tensor(N * 200)
        else:
            maxfun = tensor(float('inf'))

    one2np1 = list(range(1, N + 1))
    fsim = torch.zeros((N + 1,), dtype=torch.float32)

    for k in range(N + 1):
        fsim[k] = func(sim[k])

    ind = torch.argsort(fsim)
    fsim = torch.index_select(fsim, 0, ind)   # originally np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = torch.index_select(sim, 0, ind)  #originally .take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        if (torch.max(torch.flatten(sim[1:] - sim[0])) <= xatol and #originally np.ravel
                torch.max(torch.abs(fsim[0] - fsim[1:])) <= fatol):
            break

        xbar = torch.sum(sim[:-1], 0) / N #originally np.add.reduce(sim[:-1], 0) / N 
        xr = (1 + rho) * xbar - rho * sim[-1]
        fxr = func(xr)
        doshrink = 0

        if fxr < fsim[0]:
            xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
            fxe = func(xe)

            if fxe < fxr:
                sim[-1] = xe
                fsim[-1] = fxe
            else:
                sim[-1] = xr
                fsim[-1] = fxr
        else:  # fsim[0] <= fxr
            if fxr < fsim[-2]:
                sim[-1] = xr
                fsim[-1] = fxr
            else:  # fxr >= fsim[-2]
                # Perform contraction
                if fxr < fsim[-1]:
                    xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                    fxc = func(xc)

                    if fxc <= fxr:
                        sim[-1] = xc
                        fsim[-1] = fxc
                    else:
                        doshrink = 1
                else:
                    # Perform an inside contraction
                    xcc = (1 - psi) * xbar + psi * sim[-1]
                    fxcc = func(xcc)

                    if fxcc < fsim[-1]:
                        sim[-1] = xcc
                        fsim[-1] = fxcc
                    else:
                        doshrink = 1

                if doshrink:
                    for j in one2np1:
                        sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                        fsim[j] = func(sim[j])

        ind = torch.argsort(fsim)
        sim = torch.index_select(sim, 0, ind)
        fsim = torch.index_select(fsim, 0, ind)
        if callback is not None:
            callback(sim[0])
        iterations += 1
        if retall:
            allvecs.append(sim[0])

    x = sim[0]
    fval = torch.min(fsim)
    warnflag = 0

    if fcalls[0] >= maxfun:
        warnflag = 1
        if disp:
            print('Warning: ' + "exceeded maxfun function calls")
    elif iterations >= maxiter:
        warnflag = 2
        if disp:
            print('Warning: ' + "exceeded maxiter")
    else:
        if disp:
            print("Success!")
            print("         Current function value: %f" % fval)
            print("         Iterations: %d" % iterations)
            print("         Function evaluations: %d" % fcalls[0])

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message=msg, x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result


class OptimizeResult(dict):
    """ Represents the optimization result.
    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    success : bool
        Whether or not the optimizer exited successfully.
    status : int
        Termination status of the optimizer. Its value depends on the
        underlying solver. Refer to `message` for details.
    message : str
        Description of the cause of the termination.
    fun, jac, hess: ndarray
        Values of objective function, its Jacobian and its Hessian (if
        available). The Hessians may be approximations, see the documentation
        of the function in question.
    hess_inv : object
        Inverse of the objective function's Hessian; may be an approximation.
        Not available for all solvers. The type of this attribute may be
        either np.ndarray or scipy.sparse.linalg.LinearOperator.
    nfev, njev, nhev : int
        Number of evaluations of the objective functions and of its
        Jacobian and Hessian.
    nit : int
        Number of iterations performed by the optimizer.
    maxcv : float
        The maximum constraint violation.
    Notes
    -----
    There may be additional attributes not listed above depending of the
    specific solver. Since this class is essentially a subclass of dict
    with attribute accessors, one can see which attributes are available
    using the `keys()` method.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __repr__(self):
        if self.keys():
            m = max(map(len, list(self.keys()))) + 1
            return '\n'.join([k.rjust(m) + ': ' + repr(v)
                              for k, v in sorted(self.items())])
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())