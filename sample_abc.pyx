# ************* Sample-based Approximate Bayesian computation ************

cpdef double[:,:] sample_abc(
                           Simulator simu,
                           double[:] fixed_params,
                           double[:] observed,
                           Distance distance,
                           distribs,
                           int n_output,
                           double epsilon,
                           int n_samples,
                           double[:] init_guess,
                           double sd,
                           bool symmetric_proposal = True
                           ):
    """
    "Sample" MCMC-ABC sampler.
    Inputs:
    - simu: an instance of a Simulator class
    - fixed_params: constant parameter for the simulator
    - observed: a vector of observations
    - distance: instance of distance class
    - distribs: list of parameter distributions
    - n_output: number of output samples
    - epsilon: tolerance in acceptance criterion
    - n_samples: number of samples to base the acceptance criterion on
    - init_guess: guess
    - sd: standard deviation of the kernel
    - symmetric_proposal: whether the kernel is symmetric
    """
    cdef int n_params = len(distribs)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj, kk
    cdef double[:] params_prop = np.empty(n_params)
    cdef double[:] simulated = np.empty_like(observed)

    cdef double accprob

    cdef double[:,:] result = np.empty((n_output, n_params))

    cdef int acc_counter = 0
    cdef bool accept_MH = True
    cdef bool accept_samples

    for ii in range(1, n_output):
        accept_samples = True

        for jj in range(n_params):
            params_prop[jj] = distribs[jj].rvs(result[ii-1, jj], sd)

        simulated = simu.run(params_prop, fixed_params, n_simu)

        if not symmetric_proposal:  # no need to evaluate the MH-ratio
            accprob = 1.
            for jj in range(n_params):
                accprob *= ( distribs[jj].pdf(result[ii-1, jj], params_prop[jj], sd)
                           / distribs[jj].pdf(params_prop[jj], result[ii-1, jj], sd) )
            accept_MH = accprob >= runif()

        result[ii, :] = result[ii-1, :]
        if accept_MH:
            for jj in range(n_samples):
                kk = int( runif() * n_simu )
                if abs(observed[kk] - simulated[kk]) > epsilon:
                    accept_samples = False
                    break

            if accept_samples:
                result[ii, :] = params_prop
                acc_counter += 1

    print "Sample-ABC accepted {:.3f}% of proposals".format(100. * acc_counter / n_output)

    return result
