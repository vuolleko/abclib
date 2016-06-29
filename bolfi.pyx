include "gauss_proc.pyx"
include "minimize.pyx"
from scipy.optimize import fmin_l_bfgs_b

def abc_bolfi(
              Simulator simu,
              double[:] observed,
              list params_init,
              int n_eval,
              np.ndarray[np.float_t] limits_min,
              np.ndarray[np.float_t] limits_max,
              np.ndarray[np.float_t] hyperp_min,
              np.ndarray[np.float_t] hyperp_max,
              Distance distance,
              GaussProc gp,
              int hp_learn_interval = 20,
              int print_iter = 10,
              double sigma_jitter = 0.1
              ):
    """
    The Bayesian optimization for likelihood-free inference (BOLFI) framework.

    Following the approach described in:
    Michael U Gutmann and Jukka Corander: Bayesian optimization for
    likelihood-free inference of simulator-based statistical models,
    arXiv preprint arXiv:1501.03291, 2015.

    Inputs:
    - simu: instance of the Simulator class
    - observed: vector of observations
    - params_init: list of initial parameter vectors
    - n_eval: number of output samples
    - limits_min: vector of minimum values for parameters
    - limits_max: vector of maximum values for parameters
    - hyperp_min: vector of minimum values for GP hyperparameters
    - hyperp_max: vector of maximum values for GP hyperparameters
    - distance: instance of the Distance class
    - gp: instance of the GaussProc class (GP)
    - hp_learn_interval: interval for learning the GP hyperparameters
    - print_iter: report progress every i iterations over populations
    - sigma_jitter: factor of N(0,1) to add to found parameters
    """
    cdef int n_params = len(limits_min)
    cdef int n_simu = observed.shape[0]
    cdef int n_init = len(params_init)
    cdef int ii, jj
    cdef double[:] simulated = np.empty_like(observed)

    cdef double[:] params = np.empty(n_params)
    cdef double[:] hyperparams = np.empty(len(hyperp_min))
    cdef double[:] hyperparams_old
    cdef double distance_

    ii = 0
    jj = 0
    while ii < n_eval:

        # use initial set of parameters
        if ii < n_init:
            params = params_init[ii]

        # get rest by minimizing the acquisition function
        else:
            # optimize hyperparameters
            if (ii+1) % hp_learn_interval == 0:
                print "Optimizing hyperparameters..."
                hyperparams_old = gp.get_hyperparams()
                hyperparams = minimize_DIRECT(gp.neg_log_marginal_lh, hyperp_min, hyperp_max)
                gp.set_hyperparams(hyperparams)
                print "New hp learnt:", np.asarray(hyperparams)

            # set covariances, mean and some terms based on the new params (and hyperp)
            gp.update()
            gp.precalculate_terms()

            # check if new hyperp. are feasible (too slow to check online)
            if (ii+1) % hp_learn_interval == 0:
                if not is_pos_def( gp.covariances[:ii, :ii] ):
                    print "Resulted in non. pos. def. covariance matrix, " \
                          "reverting to previous hyperparameters."
                    gp.set_hyperparams(hyperparams_old)

            if not is_pos_def( gp.covariances[:ii, :ii] ):
                jj += 1
                if jj % print_iter == 0:
                    print "Cov. matrix not positive definite. Retrying... " \
                          "{:d}/{:d} done".format(ii, n_eval)
                ii -= 1  # discard latest
                gp.ii_eval -= 1
                gp.ii_evidence -= 1
                gp.precalculate_terms()
            else:
                jj = 0

            # find params that minimize the acquisition function
            params = fmin_l_bfgs_b(gp.acquis_fun, params, fprime=gp.grad_acquis_fun,
                                   bounds=zip(limits_min, limits_max))[0]
            # params = minimize_l_bfgs_b(gp.acquis_fun, gp.grad_acquis_fun, params,
                                       # limits_min, limits_max)

            # jitter params for more exploration
            params = params + np.random.randn(n_params) * sigma_jitter

            # limit to bounds
            params = np.clip(params, limits_min, limits_max)

            if (ii+1) % print_iter == 0:
                print "{:d}/{:d} done".format(ii+1, n_eval)

        # run the simulator for the new parameters
        simulated = simu.run(params)

        distance_ = distance.get(observed, simulated)

        gp.add_evidence(params, distance_)

        ii += 1

    return np.asarray(gp.params), np.asarray(gp.responses)
