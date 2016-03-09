cpdef double[:,:] abc_seq_mc(
                             Simulator simu,
                             double[:] fixed_params,
                             double[:] observed,
                             Distance distance,
                             list sumstats,
                             list distribs,
                             int n_output,
                             double[:] init_guess,
                             double[:] sd,
                             int n_populations,
                             double p_quantile = 0.1,
                             int print_iter = 1
                             ):
    """
    Likelihood-free sequential MC sampler.
    Inputs:
    - simu: an instance of a Simulator class
    - fixed_params: constant parameter for the simulator
    - observed: a vector of observations
    - distance: instance of distance class
    - sumstats: list of instances of summary statistics class
    - distribs: list of instances of distribution class for parameters
    - n_output: number of output samples
    - init_guess: guess
    - sd: standard deviations of the kernel
    - n_populations: number of iterations over the population
    - p_quantile: criterion for automatic acceptance threshold selection
    - print_iter: report progress every i iterations over populations
    """
    cdef int n_params = len(distribs)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj, kk, tt, sel_ind
    cdef double[:] params_prop = np.empty(n_params)
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

    # initialize with basic rejection sampler, which also gives a starting epsilon
    cdef double[:,:] result
    cdef double[:] distances
    cdef double epsilon
    result, epsilon, distances = abc_reject(simu, fixed_params, observed, distance,
                                            sumstats, distribs, n_output, init_guess,
                                            sd, p_quantile=p_quantile)

    cdef double[:,:] result_old = np.empty_like(result)
    cdef double[:] weights = np.ones(n_output)
    cdef double[:] weights_cumsum = np.empty_like(weights)
    cdef double weights_sum, weights_sum1

    cdef int acc_counter = 0
    cdef int acc_counter_pop
    cdef double randsel

    for tt in range(2, n_populations+1):
        acc_counter_pop = 0
        result_old = result.copy()

        # get new epsilon
        # epsilon = quantile(distances, p_quantile)
        # epsilon /= 2.

        # cumulative sum of weights
        weights_sum = sum_of(weights)  # for normalization
        print weights_sum
        weights_cumsum[0] = weights[0]
        for jj in range(1, n_output):
            weights[jj] /= weights_sum
            weights_cumsum[jj] = weights_cumsum[jj-1] + weights[jj]

        # update variance of proposals
        for jj in range(n_params):
            sd[jj] = 2. * sqrt( weighted_var_of( result[:, jj], weights ) )
            print sd[jj]
        print epsilon

        for ii in range(1, n_output):

            while True:

                # pick basis for proposals by sampling earlier result using their weights
                sel_ind = 0
                randsel = runif()
                while (randsel > weights_cumsum[sel_ind]):
                    sel_ind += 1

                for jj in range(n_params):
                    params_prop[jj] = (<Distribution>distribs[jj]).rvs(result_old[sel_ind, jj], sd[jj])

                simulated = simu.run(params_prop, fixed_params, n_simu)

                if n_sumstats > 0:
                    for kk in range(n_sumstats):
                        sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
                else:
                    sim_ss[:] = simulated

                acc_counter += 1
                acc_counter_pop += 1
                distances[ii] = distance.get(obs_ss, sim_ss)
                if (distances[ii] < epsilon):
                    break

            result[ii, :] = params_prop

            # set new weight, unnormalized
            weights[ii] = 1.
            weights_sum = 0.

            for kk in range(n_output):
                weights_sum1 = weights[kk]
                for jj in range(n_params):
                    weights_sum1 *= (<Distribution>distribs[jj]).pdf(result[ii, jj], result_old[kk, jj], sd[jj])
                weights_sum += weights_sum1

            for jj in range(n_params):
                weights[ii] *= (<Distribution>distribs[jj]).pdf(result[ii, jj], init_guess[jj], sd[jj])

            weights[ii] /= weights_sum

        print "{} iterations over populations done, {:.3f}% accepted".format(tt, 100. * n_output / acc_counter_pop)

    print "ABC-SEQ-MC accepted altogether {:.3f}% of proposals".format(100. * n_output / acc_counter)

    return result
