# ******************** Simulators *********************

cdef class Simulator:
    """
    A dummy parent class for simulators.
    """
    cpdef double[:] run(self, double[:] params, double[:] fixed_params,
                        unsigned int n_simu):
        pass


cdef class MA2(Simulator):
    """
    MA(2) simulator.
    """
    cpdef double[:] run(self, double[:] params, double[:] fixed_params,
                        unsigned int n_simu):
        cdef double[:] iids = np.empty(n_simu+2)
        cdef double[:] result = np.empty(n_simu)
        cdef unsigned int ii

        for ii in range(n_simu+2):
            iids[ii] = Normal().rvs()

        for ii in range(n_simu):
            result[ii] = iids[ii+2] + params[0] * iids[ii+1] + params[1] * iids[ii]

        return result


cdef class Ricker(Simulator):
    """
    Ricker, W. E. (1954) Stock and Recruitment Journal of the Fisheries
    Research Board of Canada, 11(5): 559-623.
    """
    cdef double capacity

    def __init__(self, double capacity=1.):
        self.capacity = capacity

    cpdef double[:] run(self, double[:] params, double[:] fixed_params,
                        unsigned int n_simu):
        """
        - params[0]: rate
        """
        cdef double[:] stock = np.empty(n_simu)
        cdef unsigned int ii

        stock[0] = 1e-6
        for ii in range(1, n_simu):
            stock[ii] = stock[ii-1] * exp(params[0] * (1. - stock[ii-1] / self.capacity))

        for ii in range(0, n_simu):
            stock[ii] = Poisson().rvs(stock[ii] * self.scaling)

        return stock


cdef class StochasticRicker(Simulator):
    """
    A Ricker model with observed stock ~ Poisson(true stock * scaling).
    """
    cdef double sd
    cdef double scaling

    def __init__(self, double sd=1., double scaling=1.):
        self.sd = sd
        self.scaling = scaling

    cpdef double[:] run(self, double[:] params, double[:] fixed_params,
                        unsigned int n_simu):
        """
        - params[0]: rate
        """
        cdef double[:] stock = np.empty(n_simu)
        cdef unsigned int ii

        stock[0] = 1e-6
        for ii in range(1, n_simu):
            stock[ii] = stock[ii-1] * exp(params[0] - stock[ii-1] +
                                          Normal().rvs(0., self.sd))

        # the observed is Poisson distributed
        for ii in range(n_simu):
            stock[ii] = Poisson().rvs(stock[ii] * self.scaling)

        return stock
