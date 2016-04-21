# denormalize function domain from [0, 1]
cdef inline double[:] _denorm_args(double[:] x, double[:] limits_min, double[:] limits_max):
    cdef int n_dim = x.shape[0]
    cdef double[:] xnew = np.empty(n_dim)
    cdef int ii
    for ii in range(x.shape[0]):
        xnew[ii] = x[ii] * (limits_max[ii] - limits_min[ii]) + limits_min[ii]
    return xnew


# ctypedef double (*fun_type)(double[:])

cdef minimize_DIRECT(fun, double[:] limits_min,
    double[:] limits_max, double min_measure=1e-6, int max_iter=100, int max_eval=5000):
    """
    Find the function global minimum with the DIRECT (DIviding RECTangles)
    algorithm. The function must be Lipschitz continuous.
    Jones et al., J. Optim. Theory Appl., 79(1), 157-181, 1993.
    """
    cdef int n_dim = limits_max.shape[0]
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
        print n_iter, n_eval, np.asarray(xmin), funmin

    return _denorm_args(xmin, limits_min, limits_max)
