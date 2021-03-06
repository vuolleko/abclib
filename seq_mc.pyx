cpdef double[:,:,:] abc_seq_mc(
                             int n_output,
                             Simulator simu,
                             double[:] observed,
                             list priors,
                             Distance distance,
                             list proposals,
                             double[:] schedule,
                             int n_populations,
                             double p_quantile = 0.5,
                             int print_iter = 100000
                             ):
    """
    Likelihood-free sequential MC sampler.

    Based on Algorithm 4 in:
    Jean-Michel Marin, Pierre Pudlo, Christian P Robert, and Robin J Ryder:
    Approximate bayesian computational methods, Statistics and Computing,
    22(6):1167–1180, 2012.

    Inputs:
    - n_output: number of output samples
    - simu: instance of the Simulator class
    - observed: vector of observations
    - priors: list of instances of the Distribution class
    - distance: instance of the Distance class
    - proposals: list of instances of the Distribution class
    - schedule: vector of acceptance thresholds (hybrid used)
    - n_populations: number of iterations over the population
    - p_quantile: criterion for automatic acceptance threshold selection (hybrid used)
    - print_iter: report progress every i iterations over populations
    """
    cdef int n_params = len(priors)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj, kk, tt, sel_ind
    cdef double[:] simulated = np.empty_like(observed)

    # initialize with basic rejection sampler, which also gives a starting epsilon
    cdef double[:,:,:] result = np.empty((n_populations, n_output, n_params))
    cdef double[:,:] result0
    cdef double[:] distances
    cdef double epsilon
    result0, epsilon, distances = abc_reject(n_output, simu, observed, priors,
                                             distance, p_quantile=p_quantile)

    result[0, :, :] = result0
    cdef double[:] weights = np.ones(n_output)
    cdef double[:] weights_cumsum = np.empty_like(weights)
    cdef double weights_sum, weights_sum1
    cdef double[:] wvar = np.empty(n_params)

    cdef int counter = 0
    cdef int counter_pop
    cdef double randsel

    for tt in range(1, n_populations):
        counter_pop = 0

        # get new epsilon
        epsilon = fmax( quantile(distances, p_quantile), schedule[tt] )

        # cumulative sum of weights
        weights_sum = sum_of(weights)  # for normalization
        weights[0] /= weights_sum
        weights_cumsum[0] = weights[0]
        for jj in range(1, n_output):
            weights[jj] /= weights_sum
            weights_cumsum[jj] = weights_cumsum[jj-1] + weights[jj]

        # update variance of proposals
        for jj in range(n_params):
            wvar[jj] = 2. * weighted_var_of( result[tt-1, :, jj], weights )

        print "Using threshold {} and variance {}".format(epsilon, np.asarray(wvar))

        for ii in range(1, n_output):

            while True:

                # pick basis for proposals by sampling earlier result using their weights
                sel_ind = 0
                randsel = runif()
                while (randsel > weights_cumsum[sel_ind]):
                    sel_ind += 1

                # propose new parameter set near the sampled one
                for jj in range(n_params):
                    result[tt, ii, jj] = (<Distribution> proposals[jj]).rvs( result[tt-1, sel_ind, jj], wvar[jj] )

                simulated = simu.run(result[tt, ii, :])

                counter += 1
                counter_pop += 1

                distances[ii] = distance.get(observed, simulated)
                if (distances[ii] < epsilon):
                    break

                if (counter_pop % print_iter) == 0:
                    print "{} iterations done".format(counter_pop)

            # set new weight, unnormalized
            weights[ii] = 1.
            weights_sum = 0.

            for kk in range(n_output):
                weights_sum1 = weights[kk]
                for jj in range(n_params):
                    weights_sum1 *= (<Distribution> proposals[jj]).pdf(
                        result[tt, ii, jj], result[tt-1, kk, jj], wvar[jj] )
                weights_sum += weights_sum1

            for jj in range(n_params):
                weights[ii] *= (<Distribution> priors[jj]).pdf0(result[tt, ii, jj])

            weights[ii] /= weights_sum

        print "{} iterations over populations done, {:.3f}% accepted".format(tt+1, 100. * n_output / counter_pop)

    print "ABC-SEQ-MC accepted altogether {:.3f}% of proposals".format(100. * n_output / counter)

    return np.asarray(result)
