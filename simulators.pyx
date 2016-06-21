# ******************** Simulators *********************

cdef class Simulator:
    """
    A dummy parent class for simulators.
    """
    # (cdef int n_simu, n_samples) defined in .pxd
    def __cinit__(self, int n_simu, list sumstats=[],
                  int n_samples=1, int normalized=0):
        """
        Initializer for the simulator class.
        Inputs:
        - n_simu: length of output directly from simulator
        - sumstats: list of instances of the SummaryStat class
        - n_samples: number of repeated simulator runs
        - normalized: whether to normalize simulator output (before sumstats)
        """
        self.n_simu = n_simu
        self.n_samples = n_samples
        self.normalized = normalized
        self.n_sumstats = len(sumstats)
        self.sumstats = sumstats

    def __call__(self, double[:] params):
        return self.run(params)

    cdef double[:] run(self, double[:] params):
        """
        Run the simulator and take sumstats.
        Average over sumstats if needed.
        """
        cdef double[:] simu_output = np.empty(self.n_simu)
        cdef double[:,:] all_res
        cdef double[:] result
        cdef int ii, jj

        # init arrays with proper size
        if self.n_sumstats > 0:
            all_res = np.empty((self.n_samples, self.n_sumstats))
            result = np.empty(self.n_sumstats)
        else:
            all_res = np.empty((self.n_samples, self.n_simu))
            result = np.empty(self.n_simu)

        # Run simulator for n_samples
        for ii in range(self.n_samples):
            simu_output = self.run1(params)

            if self.normalized:
                normalize(simu_output)

            if self.n_sumstats > 0:  # apply summary statistics
                result = np.array([
                    ( <SummaryStat> self.sumstats[jj] ).get(simu_output)
                                   for jj in range(self.n_sumstats) ])
                all_res[ii, :] = result

            else:  # use the result from simulator directly
                all_res[ii, :] = simu_output

        # Take mean over results
        if self.n_samples == 1:
            result = all_res[0, :]
        else:
            for ii in range(all_res.shape[1]):
                result[ii] = mean_of(all_res[:, ii])

        return result

    cdef double[:] run1(self, double[:] params):
        pass

    def from_obs(self, double[:] observed):
        """
        Calculate summary statistics for a set of observations.
        """
        cdef int jj
        cdef result
        result = np.array([
                    ( <SummaryStat> self.sumstats[jj] ).get(observed)
                           for jj in range(self.n_sumstats) ])
        return result


cdef class Simu_Gauss(Simulator):
    """
    Gaussian simulator.
    """
    cdef double[:] run1(self, double[:] params):
        cdef double[:] result = np.empty(self.n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(self.n_simu):
            result[ii] = params[0] + norm.rvs0() * params[1]

        return result


cdef class Simu_Gauss_mu(Simu_Gauss):
    """
    Gaussian simulator with sigma = 1.
    """
    cdef double[:] run1(self, double[:] params):
        return Simu_Gauss.run1(self, np.array([params[0], 1.]) )


cdef class Simu_Simple_Fun(Simulator):
    """
    Non-stochastic 1D test function.
    """
    cdef double[:] run1(self, double[:] params):
        cdef double[:] result = np.empty(self.n_simu)
        cdef int ii

        for ii in range(self.n_simu):
            result[ii] = cos(params[ii])

        return result


cdef class MA1(Simulator):
    """
    MA(1) simulator.
    """
    cdef double[:] run1(self, double[:] params):
        cdef double[:] iids = np.empty(self.n_simu+1)
        cdef double[:] result = np.empty(self.n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(self.n_simu+1):
            iids[ii] = norm.rvs0()

        for ii in range(self.n_simu):
            result[ii] = iids[ii+1] + params[0] * iids[ii]

        return result


cdef class MA2(Simulator):
    """
    MA(2) simulator.
    """
    cdef double[:] run1(self, double[:] params):
        cdef double[:] iids = np.empty(self.n_simu+2)
        cdef double[:] result = np.empty(self.n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(self.n_simu+2):
            iids[ii] = norm.rvs0()

        for ii in range(self.n_simu):
            result[ii] = iids[ii+2] + params[0] * iids[ii+1] + params[1] * iids[ii]

        return result


cdef class Ricker(Simulator):
    """
    Ricker, W. E. (1954) Stock and Recruitment Journal of the Fisheries
    Research Board of Canada, 11(5): 559-623.
    """
    cdef double[:] run1(self, double[:] params):
        """
        - params[0]: rate
        """
        cdef double[:] stock = np.empty(self.n_simu)
        cdef int ii

        stock[0] = 2.
        for ii in range(1, self.n_simu):
            stock[ii] = params[0] * stock[ii-1] * exp( -stock[ii-1] )

        return stock


cdef class StochasticRicker(Simulator):
    """
    A Ricker model with observed stock ~ Poisson(true stock * scaling).
    """
    cdef double sd
    cdef double scaling

    cdef double[:] run1(self, double[:] params):
        """
        - params[0]: log rate
        - params[1]: standard deviation of innovations
        - params[2]: scaling of the expected value from Poisson
        """
        cdef double[:] stock = np.empty(self.n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)
        cdef Poisson pois = Poisson().__new__(Poisson)

        stock[0] = 2.
        for ii in range(1, self.n_simu):
            stock[ii] = stock[ii-1] * exp(params[0] - stock[ii-1] + norm.rvs(0., params[1]))

        # the observed stock is Poisson distributed
        for ii in range(self.n_simu):
            stock[ii] = pois.rvs(stock[ii] * params[2], 1.)

        return stock
