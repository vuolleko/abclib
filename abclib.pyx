#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np
import scipy.signal
cimport numpy as np
cimport cython

from numpy.math cimport INFINITY

# some stuff from C libraries
from cpython cimport bool  # Python type
from libc.stdlib cimport rand, RAND_MAX, srand

cdef extern from "time.h":
    ctypedef int time_t
    time_t time(time_t*) nogil

cdef extern from "math.h":
    double sqrt(double x) nogil
    double exp(double x) nogil
    double log(double x) nogil
    double cos(double x) nogil
    double fmin(double x, double y) nogil
    double fmax(double x, double y) nogil
    double pow(double x, double y) nogil

# define constants
cdef double PI = np.pi


cpdef void init_rand():
    """
    Initialize the pseudo random number generator.
    """
    srand( time(NULL) )


@cython.profile(False)
cdef inline double runif() nogil:
    """
    Generates a random number in the range [0,1]
    """
    return 1. * rand() / RAND_MAX


include "auxiliary.pyx"
include "distributions.pyx"
include "similarity.pyx"
include "simulators.pyx"
include "classification.pyx"


# ************* Approximate Bayesian computation ************
include "seq_mc.pyx"


cpdef tuple abc_reject(
                       int n_output,
                       Simulator simu,
                       double[:] observed,
                       list priors,
                       Distance distance,
                       list sumstats,
                       double p_quantile = 0.1,
                       int print_iter = 100000
                       ):
    """
    Likelihood-free rejection sampler with automatic selection of acceptance threshold.
    Inputs:
    - n_output: number of output samples
    - simu: instance of the Simulator class
    - observed: vector of observations
    - priors: list of instances of the Distribution class
    - distance: instance of the Distance class
    - sumstats: list of instances of the SummaryStat class
    - p_quantile: criterion for automatic acceptance threshold selection
    - print_iter: report progress every i iterations
    """
    cdef int n_params = len(priors)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj, kk
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

    cdef double[:,:] result = np.empty((n_output, n_params))

    # find a suitable acceptance threshold epsilon
    cdef double[:] distances = np.empty(n_output)

    for ii in range(n_output):
        for jj in range(n_params):
            params_prop[jj] = (<Distribution> priors[jj]).rvs0()

        simulated = simu.run(params_prop, n_simu)

        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        result[ii, :] = params_prop
        distances[ii] = distance.get(obs_ss, sim_ss)

    cdef double epsilon = quantile(distances.copy(), p_quantile)

    cdef int counter = n_output - 1

    # use this epsilon to find rest of the samples
    for ii in range(n_output):
        if distances[ii] >= epsilon:  # otherwise already a good sample

            while True:
                for jj in range(n_params):
                    params_prop[jj] = (<Distribution> priors[jj]).rvs0()

                simulated = simu.run(params_prop, n_simu)

                if n_sumstats > 0:
                    for kk in range(n_sumstats):
                        sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
                else:
                    sim_ss[:] = simulated

                counter += 1
                distances[ii] = distance.get(obs_ss, sim_ss)
                if (distances[ii] < epsilon):
                    break

                if (counter % print_iter) == 0:
                    print "{} iterations done".format(counter)

            result[ii, :] = params_prop

    print "ABC-Reject accepted {:.3f}% of {} proposals with epsilon = {:.3g}".format(100. * n_output / counter, counter, epsilon)

    return result, epsilon, distances


cpdef double[:,:] abc_mcmc(
                           int n_output,
                           Simulator simu,
                           double[:] observed,
                           list priors,
                           Distance distance,
                           list sumstats,
                           list proposals,
                           double[:] init_guess,
                           double epsilon,
                           bool symmetric_proposal = False,
                           int print_iter = 10000
                           ):
    """
    Likelihood-free MCMC sampler.
    Inputs:
    - n_output: number of output samples
    - simu: instance of the Simulator class
    - observed: vector of observations
    - priors: list of instances of the Distribution class
    - distance: instance of the Distance class
    - sumstats: list of instances of the SummaryStat class
    - proposals: list of instances of the Distribution class (Markov kernels)
    - init_guess: guess
    - epsilon: acceptance threshold
    - symmetric_proposal: whether the proposal distribution is symmetric
    - print_iter: report progress every i iterations
    """
    cdef int n_params = len(priors)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj, kk
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

    cdef double accprob

    cdef double[:,:] result = np.empty((n_output, n_params))
    result[0, :] = init_guess

    cdef int acc_counter = 0
    cdef bool accept_MH = True

    for ii in range(1, n_output):

        # propose new parameters
        for jj in range(n_params):
            result[ii, jj] = (<Distribution> proposals[jj]).rvs1(result[ii-1, jj])

        accprob = 1.
        if symmetric_proposal:  # no need to evaluate the MH-ratio, but check that prior > 0
            for jj in range(n_params):
                accprob *= (<Distribution> priors[jj]).pdf0(result[ii, jj])
            accept_MH = accprob > 0.

        else:  # evaluate the Metropolis--Hastings ratio
            for jj in range(n_params):
                accprob *= (  (<Distribution> priors[jj]).pdf0(result[ii, jj])
                            * (<Distribution> proposals[jj]).pdf1( result[ii-1, jj], result[ii, jj] )
                            /((<Distribution> priors[jj]).pdf0(result[ii-1, jj])
                            * (<Distribution> proposals[jj]).pdf1( result[ii, jj], result[ii-1, jj] ) )
                           )
            accept_MH = accprob >= runif()

        if not accept_MH:  # no need to proceed further, reject proposal
            result[ii, :] = result[ii-1, :]

        else:  # run simulator with the proposed parameters
            simulated = simu.run(result[ii, :], n_simu)
            if n_sumstats > 0:
                for kk in range(n_sumstats):
                    sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
            else:
                sim_ss[:] = simulated

            if (distance.get(obs_ss, sim_ss) < epsilon):
                acc_counter += 1
            else:
                result[ii, :] = result[ii-1, :]

        if (ii % print_iter) == 0:
            print "{} iterations done, {} accepted so far ({:.3}%)".format(
                  ii, acc_counter, 100. * acc_counter / ii )

    print "ABC-MCMC accepted {:.3f}% of proposals".format(100. * acc_counter / n_output)

    return result
