# ****************** Probability distributions ***************


cdef class Distribution:
    """
    A dummy parent class for probability distributions.
    """
    cdef double rvs(self, double loc=0, double scale=1) nogil:
        pass

    cdef double pdf(self, double x, double loc=0, double scale=1) nogil:
        pass


cdef class Uniform(Distribution):
    cdef double rvs(self, double loc=0, double scale=1) nogil:
        """
        Returns x ~ Unif(loc, scale).
        """
        return loc + (scale - loc) * runif()

    cdef double pdf(self, double x, double loc=0, double scale=1) nogil:
        """
        Returns pdf(x) for x ~ Unif(loc, scale).
        """
        return 1. / (scale - loc)


cdef class Normal(Distribution):
    cdef double rvs(self, double loc=0, double scale=1) nogil:
        """
        Returns x ~ N(loc, scale).
        """
        # Box-Muller transform
        cdef double n01 = sqrt(-2. * log(runif())) * cos(2. * PI * runif())
        return loc + scale * n01

    cdef double pdf(self, double x, double loc=0, double scale=1) nogil:
        """
        Returns pdf(x) for x ~ N(loc, scale).
        """
        cdef double div = 1. / (scale * sqrt(2.))
        return div / sqrt(PI) * exp( -(div*(x - loc)) ** 2.)


cdef class Poisson(Distribution):
    cdef double rvs(self, double loc=1, double scale=1) nogil:
        """
        Returns x ~ Poisson(loc).
        The inherited parameter 'scale' is unused.
        Knuth's algorithm.
        """
        cdef int ii
        cdef int kk = 0
        cdef double ll = exp(-loc)
        cdef double pp = 1.

        while (pp > ll):
            pp *= runif()
            kk += 1

        return kk - 1

    cdef double pdf(self, double x, double loc=0, double scale=1) nogil:
        """
        Returns pdf(x) for x ~ Poisson(loc).
        The inherited parameter 'scale' is unused.
        """
        cdef int factorial = 1
        cdef int ii
        cdef int kk = int(x)
        cdef double powered = loc

        for ii in range(2, kk+1):
            factorial *= ii
            powered *= powered

        return powered / factorial * exp(-loc)
