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


include "distributions.pyx"
include "similarity.pyx"
include "simulators.pyx"
include "classification.pyx"


# ************* Approximate Bayesian computation ************

cpdef double[:,:] abc_reject(
                             Simulator simu,
                             double[:] fixed_params,
                             double[:] observed,
                             Distance distance,
                             list sumstats,
                             list distribs,
                             int n_output,
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
    - sumstats: list of instances of summary statistics class
    - distribs: list of instances of distribution class for parameters
    - n_output: number of output samples
    - epsilon: tolerance in acceptance criterion
    - init_guess: guess
    - sd: standard deviation of the kernel
    """
    cdef int n_params = len(distribs)
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
    result[0, :] = init_guess

    cdef int acc_counter = 0

    for ii in range(1, n_output):
        while True:
            for jj in range(n_params):
                params_prop[jj] = (<Distribution>distribs[jj]).rvs(result[ii-1, jj], sd)

            simulated = simu.run(params_prop, fixed_params, n_simu)

            if n_sumstats > 0:
                for kk in range(n_sumstats):
                    sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
            else:
                sim_ss[:] = simulated

            acc_counter += 1
            if (distance.get(obs_ss, sim_ss) < epsilon):
                break

        result[ii, :] = params_prop

    print "ABC-Reject accepted {:.3f}% of {} proposals".format(100. * (n_output-1) / acc_counter, acc_counter)

    return result


cpdef double[:,:] abc_mcmc(
                           Simulator simu,
                           double[:] fixed_params,
                           double[:] observed,
                           Distance distance,
                           list sumstats,
                           list distribs,
                           int n_output,
                           double epsilon,
                           double[:] init_guess,
                           double sd,
                           bool symmetric_proposal = True,
                           int print_iter = 10000
                           ):
    """
    Likelihood-free MCMC sampler.
    Inputs:
    - simu: an instance of a Simulator class
    - fixed_params: constant parameter for the simulator
    - observed: a vector of observations
    - distance: instance of distance class
    - sumstats: list of instances of summary statistics class
    - distribs: list of instances of distribution class for parameters
    - n_output: number of output samples
    - epsilon: tolerance in acceptance criterion
    - init_guess: guess
    - sd: standard deviation of the kernel
    - symmetric_proposal: whether the kernel is symmetric
    - print_iter: report progress every i iterations
    """
    cdef int n_params = len(distribs)
    cdef int n_simu = observed.shape[0]
    cdef int ii, jj
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

    cdef double accprob

    cdef double[:,:] result = np.empty((n_output, n_params))
    result[0, :] = init_guess

    cdef int acc_counter = 0
    cdef bool accept_MH = True

    for ii in range(1, n_output):
        for jj in range(n_params):
            params_prop[jj] = (<Distribution>distribs[jj]).rvs(result[ii-1, jj], sd)

        simulated = simu.run(params_prop, fixed_params, n_simu)

        if n_sumstats > 0:
            for kk in range(n_sumstats):
                sim_ss[kk] = (<SummaryStat> sumstats[kk]).get(simulated)
        else:
            sim_ss[:] = simulated

        if not symmetric_proposal:  # no need to evaluate the MH-ratio
            accprob = 1.
            for jj in range(n_params):
                accprob *= ( (<Distribution>distribs[jj]).pdf( result[ii-1, jj],
                             params_prop[jj], sd )
                           / (<Distribution>distribs[jj]).pdf( params_prop[jj],
                             result[ii-1, jj], sd ) )
            accept_MH = accprob >= runif()

        if (accept_MH and distance.get(obs_ss, sim_ss) < epsilon):
            result[ii, :] = params_prop
            acc_counter += 1
        else:
            result[ii, :] = result[ii-1, :]

        if (ii % print_iter) == 0:
            print "{} iterations done, {} accepted so far ({:.3}%)".format(ii, acc_counter,
                  100. * acc_counter / ii)

    print "ABC-MCMC accepted {:.3f}% of proposals".format(100. * acc_counter / n_output)

    return result
