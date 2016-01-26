#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: nonecheck=False
#cython: profile=False

import numpy as np
cimport numpy as np
cimport cython

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

# define constants
cdef double PI = np.pi


@cython.profile(False)
cdef inline double runif():
    """
    Generates a random number in the range [0,1]
    """
    return 1. * rand() / RAND_MAX


include "distributions.pyx"
include "similarity.pyx"
include "simulators.pyx"


# ************* Approximate Bayesian computation ************

cpdef double[:,:] abc_reject(
                             Simulator simu,
                             double[:] fixed_params,
                             double[:] observed,
                             Distance distance,
                             sumstats,
                             distribs,
                             unsigned int n_output,
                             double epsilon,
                             double[:] init_guess,
                             double sd
                             ):
    """
    Likelihood-free recection sampler.
    Inputs:
    - simu: an instance of a Simulator class
    - fixed_params: constant parameter for the simulator
    - observed: a vector of observations
    - distance: instance of distance class
    - sumstats: list of instances of summary statistics classes
    - distribs: list of parameter distributions
    - n_output: number of output samples
    - epsilon: tolerance in acceptance criterion
    - init_guess: guess
    - sd: standard deviation of the kernel
    """
    cdef unsigned int n_params = len(distribs)
    cdef unsigned int n_simu = observed.shape[0]
    cdef unsigned int ii, jj, kk
    cdef double[:] params_prop = np.empty(n_params)
    cdef double[:] simulated = np.empty_like(observed)

    cdef double[:] norm_observed = observed.copy()
    normalize(norm_observed)

    cdef unsigned int n_sumstats = len(sumstats)
    cdef double[:] obs_ss = np.array([sumstats[ii](norm_observed) for ii in range(n_sumstats)])
    cdef double[:] sim_ss = np.empty(n_sumstats)

    cdef double[:,:] result = np.empty((n_output, n_params))
    result[0, :] = init_guess

    cdef int acc_counter = 0

    srand(time(NULL))  # init pseudo random number generator

    for ii in range(1, n_output):
        while True:
            for jj in range(n_params):
                params_prop[jj] = distribs[jj].rvs(result[ii-1, jj], sd)

            simulated = simu.run(params_prop, fixed_params, n_simu)
            normalize(simulated)

            for kk in range(n_sumstats):
                sim_ss[kk] = sumstats[kk](simulated)

            acc_counter += 1
            if (distance(sim_ss, obs_ss) < epsilon):
                break

        result[ii, :] = params_prop

    print "ABC-Reject accepted {:.3f}% of {} proposals".format(100. * n_output / acc_counter, acc_counter)

    return result


cpdef double[:,:] abc_mcmc(
                           Simulator simu,
                           double[:] fixed_params,
                           double[:] observed,
                           Distance distance,
                           sumstats,
                           distribs,
                           unsigned int n_output,
                           double epsilon,
                           double[:] init_guess,
                           double sd,
                           bool symmetric_proposal = True
                           ):
    """
    Likelihood-free MCMC sampler.
    Inputs:
    - simu: an instance of a Simulator class
    - fixed_params: constant parameter for the simulator
    - observed: a vector of observations
    - distance: instance of distance class
    - sumstats: list of instances of summary statistics classes
    - distribs: list of parameter distributions
    - n_output: number of output samples
    - epsilon: tolerance in acceptance criterion
    - init_guess: guess
    - sd: standard deviation of the kernel
    - symmetric_proposal: whether the kernel is symmetric
    """
    cdef unsigned int n_params = len(distribs)
    cdef unsigned int n_simu = observed.shape[0]
    cdef unsigned int ii, jj
    cdef double[:] params_prop = np.empty(n_params)
    cdef double[:] simulated = np.empty_like(observed)

    cdef double[:] norm_observed = observed.copy()
    normalize(norm_observed)

    cdef unsigned int n_sumstats = len(sumstats)
    cdef double[:] obs_ss = np.array([sumstats[ii](norm_observed) for ii in range(n_sumstats)])
    cdef double[:] sim_ss = np.empty(n_sumstats)
    cdef double accprob

    cdef double[:,:] result = np.empty((n_output, n_params))

    # initial guess from ABC rejection sampler
    result[0, :] = abc_reject(simu, fixed_params, observed, distance, sumstats,
                              distribs, 2, epsilon/10., init_guess, sd)[1, :]

    cdef int acc_counter = 0
    cdef bool accept_MH = True

    srand(time(NULL))  # init pseudo random number generator

    for ii in range(1, n_output):
        for jj in range(n_params):
            params_prop[jj] = distribs[jj].rvs(result[ii-1, jj], sd)

        simulated = simu.run(params_prop, fixed_params, n_simu)
        normalize(simulated)

        for kk in range(n_sumstats):
            sim_ss[kk] = sumstats[kk](simulated)

        if not symmetric_proposal:  # no need to evaluate the MH-ratio
            accprob = 1.
            for jj in range(n_params):
                accprob *= ( distribs[jj].pdf(result[ii-1, jj], params_prop[jj], sd)
                           / distribs[jj].pdf(params_prop[jj], result[ii-1, jj], sd) )
            accept_MH = accprob >= runif()

        if (accept_MH and distance(sim_ss, obs_ss) < epsilon):
            result[ii, :] = params_prop
            acc_counter += 1
        else:
            result[ii, :] = result[ii-1, :]

    print "ABC-MCMC accepted {:.3f}% of proposals".format(100. * acc_counter / n_output)

    return result
