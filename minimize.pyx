# denormalize function domain from [0, 1]
cdef inline double[:] _denorm_args(double[:] x, double[:] limits_min, double[:] limits_max):
    cdef int n_dim = x.shape[0]
    cdef double[:] xnew = np.empty(n_dim)
    cdef int ii
    for ii in range(x.shape[0]):
        xnew[ii] = x[ii] * (limits_max[ii] - limits_min[ii]) + limits_min[ii]
    return xnew


# ctypedef double (*fun_type)(double[:])

def minimize_DIRECT(fun, double[:] limits_min,
    double[:] limits_max, double min_measure=1e-9, int max_iter=0, int max_eval=5000):
    """
    Find the function global minimum with the DIRECT (DIviding RECTangles)
    algorithm. The function must be Lipschitz continuous.
    Jones et al., J. Optim. Theory Appl., 79(1), 157-181, 1993.
    """
    cdef int n_dim = limits_max.shape[0]
    if max_iter == 0:
        max_iter = n_dim * 100
    cdef int maxsize = max_eval * 2 * n_dim
    cdef np.ndarray[np.float_t, ndim=2] centers = np.empty((maxsize, n_dim))
    cdef np.ndarray[np.float_t] funvals = np.empty(maxsize)
    cdef np.ndarray[np.float_t, ndim=2] widths = np.empty_like(centers)
    cdef np.ndarray[np.float_t] delta = np.zeros(n_dim)

    # initialization of DIRECT
    cdef double epsilon = 1e-4
    centers[0, :] = 0.5
    funvals[0] = fun( _denorm_args(centers[0, :], limits_min, limits_max) )
    widths[0, :] = 1.
    cdef double funmin = funvals[0]
    cdef double[:] xmin = centers[0, :]
    cdef int n_eval = 1
    cdef int n_iter = 0

    cdef int ii, jj, kk, parent, ind_funmin
    cdef double max1, max2
    cdef double[:] dimmins
    cdef long[:] maxdims, inds_sorted, inds1, inds2, inds3
    cdef np.ndarray[np.float_t] measures = np.ones(1)
    cdef list potentials

    while np.min(measures) > min_measure and n_iter < max_iter and n_eval < max_eval:
        # first division
        if n_iter == 0:
            potentials = [0]

        # find potentially optimal hyper-rectangles
        else:
            potentials = []  # for indices
            measures = (widths[:n_eval]**2.).sum(axis=1)  # measure of size
            for jj in range(n_eval):

                inds3 = np.where(measures == measures[jj])[0]
                if np.any(funvals[inds3] < funvals[jj]):
                    continue  # reject

                inds2 = np.where(measures > measures[jj])[0]
                if len(inds2) > 0:

                    inds1 = np.where(measures < measures[jj])[0]
                    if len(inds1) > 0:
                        max1 = np.max( (funvals[jj] - funvals[inds1])
                                      / (measures[jj] - measures[inds1]) )
                    else:
                        max1 = -np.inf
                        # print "Empty"
                    min2 = np.min( (funvals[inds2] - funvals[jj])
                                  / (measures[inds2] - measures[jj]) )

                    if min2 < max1:
                        continue  # reject

                    min2 *= measures[jj]
                    if funmin == 0:
                        if funvals[jj] > min2:
                            continue  # reject
                    else:
                        if epsilon * abs(funmin) > funmin - funvals[jj] + min2:
                            continue  # reject

                potentials.append(jj)  # accept if we end up here
            # print "Potentials: ", potentials

        # process all potentially optimal hyper-rectangles
        for parent in potentials:

            # find new centers with the longest dims split into thirds
            maxdims = np.where( widths[parent, :] == np.max(widths[parent, :]) )[0]
            dimmins = np.empty_like(maxdims, dtype=np.float)

            for ii in range(maxdims.shape[0]):
                jj = n_eval + 2 * ii
                delta[maxdims[ii]] = widths[parent, maxdims[ii]] / 3.
                centers[jj, :] = centers[parent, :] + delta
                centers[jj+1, :] = centers[parent, :] - delta
                funvals[jj] = fun( _denorm_args(centers[jj, :], limits_min, limits_max) )
                funvals[jj+1] = fun( _denorm_args(centers[jj+1, :], limits_min, limits_max) )
                delta[maxdims[ii]] = 0.
                dimmins[ii] = min(funvals[jj], funvals[jj+1])
                widths[jj, :] = widths[parent, :]
                widths[jj+1, :] = widths[parent, :]

            # split rectanges into thirds along the largest dim(s)
            inds_sorted = np.argsort(dimmins)  # order of splitting
            for ii in range(maxdims.shape[0]):
                widths[parent, maxdims[inds_sorted[ii]]] /= 3.
                for kk in range(ii, maxdims.shape[0]):
                    jj = n_eval + 2 * inds_sorted[kk]
                    widths[jj, maxdims[inds_sorted[ii]]] /= 3.
                    widths[jj+1, maxdims[inds_sorted[ii]]] /= 3.

            n_eval += 2 * maxdims.shape[0]

        ind_funmin = np.argmin(funvals[:n_eval])
        funmin = funvals[ind_funmin]
        xmin = centers[ind_funmin, :]
        n_iter += 1
        # print n_iter, n_eval, np.asarray(xmin), funmin

    return _denorm_args(xmin, limits_min, limits_max)


def minimize_conjgrad(fun, grad, double[:] guess,
                      double tolerance=1e-9, int max_iter=1000,
                      double max_stepsize=1., double min_stepsize=1e-8,
                      double ss_factor=0.5, double c1_factor=1e-4):
    """
    Find a function minimum with the non-linear conjugate gradient method
    using the Polak-Ribiere beta coefficient. Line search is performed with
    backtracking.
    """
    cdef double[:] xmin = guess
    cdef np.ndarray[np.float_t] hilldir, hilldir_old, stepdir
    cdef double beta, stepsize, compval, funval
    cdef int ii = 0

    # initial directions
    hilldir = grad(xmin)
    stepdir = -1. * hilldir

    while not np.allclose( hilldir, 0., atol=tolerance ) and ii < max_iter:
        ii += 1
        hilldir_old = hilldir.copy()

        # Backtrack to find appropriate stepsize
        stepsize = max_stepsize
        compval = hilldir.T.dot(stepdir) * c1_factor
        funval = fun(xmin)
        while fun(xmin + stepsize * stepdir) > funval + compval * stepsize and stepsize > min_stepsize:
            stepsize *= ss_factor

        xmin = xmin + stepsize * stepdir
        hilldir = grad(xmin)

        # Polak-Ribiere
        beta = hilldir.T.dot(hilldir - hilldir_old) / hilldir_old.T.dot(hilldir_old)
        beta = max(0., beta)  # reset if not descent direction

        stepdir = beta * stepdir - hilldir

    return xmin


def minimize_l_bfgs_b(fun, grad, double[:] guess,
                      np.ndarray[np.float_t] limits_min,
                      np.ndarray[np.float_t] limits_max,
                      int max_iter=1000, double min_stepsize=1e-8,
                      double ss_factor=0.5, double c1_factor=1e-4,
                      int n_history=10, double epsilon_curvature=2.2e-16):
    """
    Find a function minimum in a constrained area with the L-BFGS-B algorithm.
    R. H. Byrd, P. Lu and J. Nocedal. A Limited Memory Algorithm for Bound Constrained
    Optimization, (1995), SIAM Journal on Scientific and Statistical Computing,
    16, 5, pp. 1190-1208.
    Line search is performed with simple backtracking (TODO: improve).
    """
    cdef int n_dim = guess.shape[0]
    cdef np.ndarray[np.float_t] xmin = np.asarray(guess)
    cdef np.ndarray[np.float_t] x_cauchy = np.zeros(n_dim)
    cdef np.ndarray[np.float_t] gradx = grad(xmin)
    cdef np.ndarray[np.float_t] x_bar, xdiff, gdiff, grad_proj, d_u, direction
    cdef np.ndarray[np.float_t] x_old, g_old, alphas_max
    cdef np.ndarray[np.float_t] tn, r_c, c, d, p, wbt, gwM, v
    cdef np.ndarray[np.float_t, ndim=2] Y, S, L, D, M, W, SY, N, WTZ
    cdef np.ndarray[np.int_t] freevars
    Y = np.empty((n_dim, 0))
    S = np.empty((n_dim, 0))
    L = np.empty((0, 0))
    D = np.empty((0, 0))
    M = np.empty((0, 0))
    W = np.empty((n_dim, 0))
    cdef double theta = 1.
    cdef list F
    cdef double stepsize, funval, compval, fp, fpp
    cdef double delta_t, delta_t_min, t, t_old, z_b, theta_inv
    cdef int b
    grad_proj = np.ones(n_dim)
    cdef double zero_epsilon = 1e-17

    cdef int kk = 0
    while np.abs(grad_proj).max() > 1e-5 and kk < max_iter:
        # print "\nIter {}, x = {}, g = {}".format(kk, xmin.ravel(), gradx.ravel())

        # ****** Calculate the Cauchy point ******
        tn = np.where(gradx < 0., (xmin-limits_max)/gradx,
                      np.where(gradx > 0., (xmin-limits_min)/gradx, np.inf) )
        d = np.where(tn < zero_epsilon, 0., -gradx)
        F = [ii for ii in np.argsort(tn) if tn[ii] > zero_epsilon]
        p = W.T.dot(d)
        c = np.zeros_like(p)

        if len(F) > 0:
            fp = -d.T.dot(d)
            fpp = -theta * fp - p.T.dot(M).dot(p)
            if abs(fpp) < zero_epsilon:
                delta_t_min = 0.
            else:
                delta_t_min = -fp / fpp
            t_old = 0.
            t = tn[ F[0] ]
            delta_t = t

            while delta_t_min >= delta_t:
                b = F.pop(0)
                t = tn[b]
                delta_t = t - t_old
                x_cauchy[b] = limits_max[b] if d[b] > 0. else limits_min[b]
                z_b = x_cauchy[b] - xmin[b]
                c += delta_t * p
                wbt = W[b, :]
                gwM = gradx[b] * wbt.dot(M)
                fp += delta_t * fpp + gradx[b]*gradx[b] + theta*gradx[b]*z_b - gwM.dot(c)
                fpp -= theta*gradx[b]*gradx[b] + 2. * gwM.dot(p) + gradx[b] * gwM.dot(wbt.T)
                p += gradx[b] * wbt.T
                d[b] = 0.
                if abs(fpp) < zero_epsilon:
                    delta_t_min = 0.
                else:
                    delta_t_min = -fp / fpp
                t_old = t

                if len(F) == 0:
                    break

            delta_t_min = max(delta_t_min, 0.)
            t_old += delta_t_min
            for ii in F:
                x_cauchy[ii] = xmin[ii] + t_old * d[ii]
            c += delta_t_min * p

        # print "Cauchy: ", x_cauchy

        # ****** Calculate the search direction (direct primal method) ******
        freevars = np.where((x_cauchy != limits_min) & (x_cauchy != limits_max))[0]
        # print "Free vars at CP: ", len(freevars)
        theta_inv = 1. / theta

        r_c = ( gradx + theta * (x_cauchy - xmin) - W.dot(M).dot(c) )[freevars]
        WTZ = W.T[:, freevars]
        v = M.dot(WTZ).dot(r_c)
        N = theta_inv * WTZ.dot(WTZ.T)
        N = np.eye(N.shape[0]) - M.dot(N)
        v = np.linalg.solve( N, v )

        d_u = -theta_inv * ( r_c + theta_inv * WTZ.T.dot(v) )
        alphas_max = np.where(d_u > 0.,
                              (limits_max[freevars] - x_cauchy[freevars]) / d_u,
                              np.where( d_u < 0.,
                              (limits_min[freevars] - x_cauchy[freevars]) / d_u,
                              0. )
                             )

        x_bar = x_cauchy.copy()
        if np.any(freevars):
            x_bar[freevars] += alphas_max.min() * d_u

        # Backtrack to find appropriate stepsize
        direction = x_bar - xmin
        stepsize = 1.
        compval = gradx.T.dot(direction) * c1_factor
        funval = fun(xmin)
        while fun(xmin + stepsize * direction) > funval + compval * stepsize and stepsize > min_stepsize:
            stepsize *= ss_factor

        # update x and calculate the gradient
        x_old = xmin.copy()
        g_old = gradx.copy()
        xmin += stepsize * direction
        gradx = grad(xmin)

        # update history and related matrices, if the curvature condition is ok
        xdiff = xmin - x_old
        gdiff = gradx - g_old

        if gdiff.dot(xdiff) > epsilon_curvature * gdiff.dot(gdiff):
            if Y.shape[1] < n_history:
                Y = np.concatenate((Y, gdiff[:, np.newaxis]), axis=1)
                S = np.concatenate((S, xdiff[:, np.newaxis]), axis=1)
            else:
                Y = np.roll(Y, -1, axis=1)
                Y[:, -1] = gdiff
                S = np.roll(S, -1, axis=1)
                S[:, -1] = xdiff

            SY = S.T.dot(Y)
            theta = gdiff.dot(gdiff) / gdiff.dot(xdiff)
            W = np.concatenate((Y, theta * S), axis=1)
            L = np.tril(SY, -1)
            D = np.diagflat(SY.diagonal())
            M = np.asarray( np.bmat( [[-D, L.T], [L, theta * S.T.dot(S) ]] ).I )

        kk += 1

        # compute the projected gradient for convergence check
        grad_proj = np.maximum( np.minimum(xmin-gradx, limits_max), limits_min) - xmin

    return xmin
