include "gauss_proc.pyx"
include "minimize.pyx"
from scipy.optimize import fmin_l_bfgs_b


def abc_bolfi(
              Simulator simu,
              double[:] observed,
              list params_init,
              int n_eval,
              double[:] limits_min,
              double[:] limits_max,
              double[:] hyperp_min,
              double[:] hyperp_max,
              Distance distance,
              list sumstats,
              GaussProc gp,
              int hp_learn_interval = 20,
              int print_iter = 10,
              double sigma_jitter = 0.1,
              double param_epsilon = 1e-2
              ):
    """
    Approach using Bayesian optimization for likelihood-free inference (BOLFI)
    Inputs:
    - simu: instance of the Simulator class
    - observed: vector of observations
    - priors: list of instances of the Distribution class
    - distance: instance of the Distance class
    - sumstats: list of instances of the SummaryStat class
    - print_iter: report progress every i iterations over populations
    """
    cdef int n_params = len(limits_min)
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
    cdef double[:] hyperparams = np.empty(len(hyperp_min))
    cdef double distance_

    for ii in range(n_eval):

        # use initial set of parameters
        if ii < n_init:
            params = params_init[ii]

        # get rest by minimizing the acquisition function
        else:
            # optimize hyperparameters
            if ii % hp_learn_interval == 0:
                hyperparams = minimize_DIRECT(gp.log_marginal_lh, hyperp_min, hyperp_max)
                # hyperparams = fmin_l_bfgs_b(gp.log_marginal_lh, hyperparams,
                  # bounds=zip(hyperp_min, hyperp_max))[0]
                gp.set_hyperparams(hyperparams)
                print "New hp learnt:", np.asarray(hyperparams)

            # set covariances, mean and some terms based on the new params
            gp.update()
            gp.precalculate_terms()

            # find params that minimize the acquisition function
            # if ii == n_init:  # avoid using boundary as initial guess
                # params = params_init[1]
            # params = minimize_conjgrad(gp.acquis_fun, gp.grad_acquis_fun, params)
            # params = minimize_DIRECT(gp.acquis_fun, limits_min, limits_max,
                                     # max_iter=200)
            params = fmin_l_bfgs_b(gp.acquis_fun, params, fprime=gp.grad_acquis_fun,
                                   bounds=zip(limits_min, limits_max))[0]

            while True:
                # jitter params for more exploration
                params = params + np.random.randn(n_params) * sigma_jitter

                # limit to bounds
                params = np.clip(params, limits_min, limits_max)

                # avoid too similar params (may cause non-pos. def. cov. matrix)
                if not np.any( np.all( np.isclose(params, gp.params, atol=param_epsilon), axis=1 ) ):
                    break
                print "A too similar parameter set skipped at ii=", ii

            if ii % print_iter == 0:
                print "{:d}/{:d} done".format(ii, n_eval)

        # run the simulator for the new parameters
        simulated = simu.run(params)
        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        distance_ = distance.get(obs_ss, sim_ss)

        gp.add_evidence(params, distance_)

    return np.asarray(gp.params), np.asarray(gp.responses)
