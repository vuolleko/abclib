include "gauss_proc.pyx"
include "min_direct.pyx"

def abc_bolfi(
              Simulator simu,
              double[:] observed,
              list params_init,
              int n_eval,
              list priors,
              double[:] limits_min,
              double[:] limits_max,
              double[:] hyperp_min,
              double[:] hyperp_max,
              Distance distance,
              list sumstats,
              GaussProc gp,
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

    cdef double[:] params = np.empty(n_params)
    cdef double distance_

    for ii in range(n_eval):

        # use initial set of parameters
        if ii < n_init:
            params = params_init[ii]

        # get rest by minimizing the acquisition function
        else:
            gp.precalculate_terms()

            # optimize hyperparameters
            hyperparams = minimize_DIRECT(gp.log_marginal_lh, hyperp_min, hyperp_max)
            gp.set_hyperparams(hyperparams)

            # find params that minimize the acquisition function
            params = minimize_DIRECT(gp.acquis_fun, limits_min, limits_max)

        # run the simulator for the new parameters
        simulated = simu.run(params)
        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        distance_ = distance.get(obs_ss, sim_ss)

        # set covariances and mean based on the new params
        gp.add_evidence(params, distance_)
        gp.update()

    return gp.params, gp.distances


