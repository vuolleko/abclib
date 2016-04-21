include "min_direct.pyx"

def abc_bolfi(
              Simulator simu,
              double[:] observed,
              list params_init,
              int n_eval,
              list priors,
              double[:] limits_min,
              double[:] limits_max,
              Distance distance,
              list sumstats,
              double p_quantile = 0.5,
              int print_iter = 10
              ):
    """
    Approach using Bayesian optimization for likelihood-free inference (BOLFI)
    Inputs:
    - n_output: number of output samples
    - simu: instance of the Simulator class
    - observed: vector of observations
    - priors: list of instances of the Distribution class
    - distance: instance of the Distance class
    - sumstats: list of instances of the SummaryStat class
    - print_iter: report progress every i iterations over populations
    """
    cdef int n_params = len(priors)
    cdef int n_simu = observed.shape[0]
    cdef int n_init = len(params_init)
    cdef int ii, jj
    cdef double[:] simulated = np.empty_like(observed)

    cdef int n_sumstats = len(sumstats)
    cdef double[:] obs_ss
    cdef double[:] sim_ss
    if n_sumstats > 0:
        obs_ss = np.array([(<SummaryStat> sumstats[ii]).get(observed)
                           for ii in range(n_sumstats)])
        sim_ss = np.empty(n_sumstats)
    else:
        obs_ss = observed
        sim_ss = np.empty(n_simu)

    cdef double sigma2_signal = 1.
    cdef double sigma2_obs = 1.
    cdef double[:] scale_signal = np.ones(n_params)
    cdef double[:] a_factor = np.ones(n_params)
    cdef double[:] b_factor = np.ones(n_params)
    cdef double c_factor = 1.
    cdef double eta_factor
    cdef double[:, :] params = np.empty((n_eval, n_params))

    cdef np.ndarray[np.float_t] distances = np.empty(n_eval)
    cdef np.ndarray[np.float_t, ndim=2] covariances = np.empty((n_eval, n_eval))
    cdef np.ndarray[np.float] means = np.empty(n_eval)

    # run the simulator for the initial set of parameters
    for ii in range(n_init):
        simulated = simu.run( params_init[ii] )
        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        distances[ii] = distance.get(obs_ss, sim_ss)

    # set covariances and means for the initial set
    for ii in range(n_init):
        params[ii, :] = params_init[ii]
        covariances[ii, ii] = sigma2_obs
        for jj in range(ii):
            covariances[ii, jj] = _sq_exp_cov(params[ii, :], params[jj, :],
                                              sigma2_signal, scale_signal)
            covariances[jj, ii] = covariances[ii, jj]  # symmetry

        means[ii] = _sum_conv_qp(params[ii, :], a_factor, b_factor, c_factor)

    # get rest of the requested evalutions by minimizing the acquisition function
    cdef double[:, :] chol_L
    # cdef np.ndarray[np.float_t, ndim=2] chol_L
    cdef double[:] chol_solved
    # cdef np.ndarray[np.float_t] chol_solved
    for ii in range(n_init, n_eval):
        chol_L = np.asarray( cholesky(covariances[:ii, :ii]) )
        chol_solved = solve_ut( chol_L.T,
                                solve_lt(chol_L, distances[:ii] - means[:ii])
                               )
        eta_factor = 2. * log( ii**(n_params/2. + 2.) * PI * PI / 0.3 )

        f = lambda x: _acquis_fun(x, params[:ii, :], chol_L, chol_solved,
                                  a_factor, b_factor, c_factor, sigma2_signal,
                                  scale_signal, eta_factor)

        # find params that minimize the acquisition function
        params[ii, :] = minimize_DIRECT(f, limits_min, limits_max)

        # set covariances and mean based on the new params
        for jj in range(ii):
            covariances[ii, jj] = _sq_exp_cov(params[ii, :], params[jj, :],
                                              sigma2_signal, scale_signal)
            covariances[jj, ii] = covariances[ii, jj]  # symmetry

        means[ii] = _sum_conv_qp(params[ii, :], a_factor, b_factor, c_factor)

        # run the simulator for the new parameters
        simulated = simu.run( params[ii, :] )
        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        distances[ii] = distance.get(obs_ss, sim_ss)

    return params, distances
    # return distances


# Covariance function: Squared exponential covariance
cdef inline double _sq_exp_cov(double[:] params1, double[:] params2,
                               double sigma2_signal, double[:] scale) nogil:
    cdef int nn = params1.shape[0]
    cdef int ii
    cdef double sum0 = 0.
    cdef double temp

    for ii in range(nn):
        temp = (params1[ii] - params2[ii]) / scale[ii]
        sum0 += temp * temp

    return sigma2_signal * exp(-sum0)


# Gradient of Covariance function: Squared exponential covariance
cdef inline double _grad_sq_exp_cov(double[:] params1, double[:] params2,
                               double sigma2_signal, double[:] scale, int ii) nogil:
    cdef double res = _sq_exp_cov(params1, params2, sigma2_signal, scale)
    res *= 2. * (params1[ii] - params2[ii]) / ( scale[ii] * scale[ii] )
    return res


# Mean function: sum of convex quadratic polynomials
cdef inline double _sum_conv_qp(double[:] params, double[:] a, double[:] b,
                                double c) nogil:
    cdef int nn = params.shape[0]
    cdef int ii
    cdef double sum0 = 0.

    for ii in range(nn):
        sum0 += a[ii] * params[ii] * params[ii] + b[ii] * params[ii] + c

    return sum0


# Gradient of mean function: sum of convex quadratic polynomials
cdef inline double _grad_sum_conv_qp(double[:] params, double[:] a,
                                     double[:] b, int ii) nogil:
    return a[ii] * params[ii] + b[ii]


# Acquisition function
cdef double _acquis_fun(np.ndarray[np.float_t] params_new,
                        double[:, :] params,
                        np.ndarray[np.float_t, ndim=2] chol_L,
                        np.ndarray[np.float_t] chol_solved,
                        double[:] a_factor,
                        double[:] b_factor,
                        double c_factor,
                        double sigma2_signal,
                        double[:] scale_signal,
                        double eta_factor
                        ):
    cdef int tt = params.shape[0]
    cdef np.ndarray[np.float_t] k_vec = np.empty(tt)
    cdef int ii
    for ii in range(tt):
        k_vec[ii] = _sq_exp_cov(params_new, params[ii, :], sigma2_signal, scale_signal)

    cdef double mean_new = _sum_conv_qp(params_new, a_factor, b_factor, c_factor)
    mean_new += k_vec.T.dot(chol_solved)
    cdef np.ndarray[np.float_t] v_vec = np.linalg.solve(chol_L, k_vec)
    cdef double var_new = sigma2_signal - v_vec.T.dot(v_vec)

    return mean_new + sqrt(eta_factor * var_new)


# Gradient of acquisition function
cdef double[:] _grad_acquis_fun(np.ndarray[np.float_t] params_new,
                        double[:, :] params,
                        np.ndarray[np.float_t, ndim=2] chol_L,
                        np.ndarray[np.float_t] chol_solved,
                        double[:] a_factor,
                        double[:] b_factor,
                        double c_factor,
                        double sigma2_signal,
                        double[:] scale_signal,
                        double eta_factor
                        ):
    cdef int nn = params.shape[1]
    cdef double[:] grads = np.empty(nn)
    cdef int tt = params.shape[0]
    cdef np.ndarray[np.float_t] k_vec = np.empty(tt)
    cdef np.ndarray[np.float_t] grad_k_vec = np.empty(tt)
    cdef int ii

    for ii in range(tt):
        k_vec[ii] = _sq_exp_cov(params_new, params[ii, :], sigma2_signal, scale_signal)
        grad_k_vec[ii] = _grad_sq_exp_cov(params_new, params[ii, :], sigma2_signal,
                                          scale_signal, ii)

    cdef np.ndarray[np.float_t] v_vec = solve_lt(chol_L, k_vec)
    cdef double var_new = sigma2_signal - v_vec.T.dot(v_vec)
    cdef np.ndarray[np.float_t] v_vec_solved = solve_ut(chol_L.T, v_vec)

    for ii in range(nn):
        grads[ii] = _grad_sum_conv_qp(params_new, a_factor, b_factor, ii)
        grads[ii] += grad_k_vec[ii] * chol_solved[ii]
        if var_new != 0:
            grads[ii] += sqrt(eta_factor / var_new) * grad_k_vec[ii] * v_vec_solved[ii]

    return grads
