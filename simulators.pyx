# ******************** Simulators *********************

cdef class Simulator:
    """
    A dummy parent class for simulators.
    """
    def __call__(self, double[:] params, double[:] fixed_params, int n_simu):
        return self.run(params, fixed_params, n_simu)

    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        pass


cdef class Gauss(Simulator):
    """
    Gaussian simulator.
    """
    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        cdef double[:] result = np.empty(n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(n_simu):
            result[ii] = params[0] + norm.rvs() * params[1]

        return result


cdef class MA1(Simulator):
    """
    MA(1) simulator.
    """
    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        cdef double[:] iids = np.empty(n_simu+1)
        cdef double[:] result = np.empty(n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(n_simu+1):
            iids[ii] = norm.rvs()

        for ii in range(n_simu):
            result[ii] = iids[ii+1] + params[0] * iids[ii]

        return result


cdef class MA2(Simulator):
    """
    MA(2) simulator.
    """
    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        cdef double[:] iids = np.empty(n_simu+2)
        cdef double[:] result = np.empty(n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)

        for ii in range(n_simu+2):
            iids[ii] = norm.rvs()

        for ii in range(n_simu):
            result[ii] = iids[ii+2] + params[0] * iids[ii+1] + params[1] * iids[ii]

        return result


cdef class Ricker(Simulator):
    """
    Ricker, W. E. (1954) Stock and Recruitment Journal of the Fisheries
    Research Board of Canada, 11(5): 559-623.
    """
    cdef double capacity

    def __cinit__(self, double capacity=1.):
        self.capacity = capacity

    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        """
        - params[0]: rate
        """
        cdef double[:] stock = np.empty(n_simu)
        cdef int ii
        cdef Poisson pois = Poisson().__new__(Poisson)

        stock[0] = 1e-6
        for ii in range(1, n_simu):
            stock[ii] = stock[ii-1] * exp(params[0] * (1. - stock[ii-1] / self.capacity))

        for ii in range(0, n_simu):
            stock[ii] = pois.rvs(stock[ii] * self.scaling)

        return stock


cdef class StochasticRicker(Simulator):
    """
    A Ricker model with observed stock ~ Poisson(true stock * scaling).
    """
    cdef double sd
    cdef double scaling

    def __cinit__(self, double sd=1., double scaling=1.):
        self.sd = sd
        self.scaling = scaling

    cdef double[:] run(self, double[:] params, double[:] fixed_params,
                        int n_simu):
        """
        - params[0]: rate
        """
        cdef double[:] stock = np.empty(n_simu)
        cdef int ii
        cdef Normal norm = Normal().__new__(Normal)
        cdef Poisson pois = Poisson().__new__(Poisson)

        stock[0] = 1e-6
        for ii in range(1, n_simu):
            stock[ii] = stock[ii-1] * exp(params[0] - stock[ii-1] +
                                          norm.rvs(0., self.sd))

        # the observed stock is Poisson distributed
        for ii in range(n_simu):
            stock[ii] = pois.rvs(stock[ii] * self.scaling)

        return stock
