# ******************** Simulators *********************

cdef class Simulator:
    """
    A dummy parent class for simulators.
    """
    # (cdef int n_simu, n_samples) defined in .pxd
    def __cinit__(self, int n_simu, int n_samples=1, int normalized=0):
        self.n_simu = n_simu
        self.n_samples = n_samples
        self.normalized = normalized

    def __call__(self, double[:] params):
        return self.run(params)

    cdef double[:] run(self, double[:] params):
        """
        Run the simulator and take average over samples if needed.
        """
        cdef double[:,:] all_res
        cdef double[:] result = np.empty(self.n_simu)
        cdef int ii

        if self.n_samples == 1:
            result = self.run1(params)

        else:  # average over several runs
            all_res = np.empty((self.n_samples, self.n_simu))
            for ii in range(self.n_samples):
                all_res[ii, :] = self.run1(params)
            for ii in range(self.n_simu):
                result[ii] = mean_of(all_res[:, ii])

        if self.normalized:
            normalize(result)
        return result

    cdef double[:] run1(self, double[:] params):
        pass


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
