# ****************** Probability distributions ***************


cdef class Distribution:
    """
    A dummy parent class for probability distributions.
    """
    cdef double loc, scale, minval, maxval
    def __cinit__(self, double loc=0., double scale=1., double minval=-np.inf,
                  double maxval=np.inf):
        self.loc = loc
        self.scale = scale
        self.minval = minval
        self.maxval = maxval

    cdef double rvs(self, double loc, double scale) nogil:
        pass

    cdef double rvs0(self) nogil:
        return self.rvs(self.loc, self.scale)

    cdef double rvs1(self, double loc) nogil:
        return self.rvs(loc, self.scale)

    cdef double pdf(self, double x, double loc, double scale) nogil:
        pass

    cdef double pdf0(self, double x) nogil:
        return self.pdf(x, self.loc, self.scale)

    cdef double pdf1(self, double x, double loc) nogil:
        return self.pdf(x, loc, self.scale)


cdef class Uniform(Distribution):
    cdef double rvs(self, double loc, double scale) nogil:
        """
        Returns x ~ Unif(loc, scale).
        """
        return loc + (scale - loc) * runif()

    cdef double pdf(self, double x, double loc, double scale) nogil:
        """
        Returns pdf(x) for x ~ Unif(loc, scale).
        """
        if (x < loc) or (x > scale):
            return 0.
        return 1. / (scale - loc)


cdef class Normal(Distribution):
    cdef double rvs(self, double loc, double scale) nogil:
        """
        Returns x ~ N(loc, scale) using Box-Muller transform.
        """
        cdef double result, n01
        while True:
            n01 = sqrt(-2. * log(runif())) * cos(2. * PI * runif())
            result = loc + scale * n01
            if (result >= self.minval) & (result <= self.maxval):
                break
        return result

    cdef double pdf(self, double x, double loc, double scale) nogil:
        """
        Returns pdf(x) for x ~ N(loc, scale).
        """
        cdef double div = 1. / (scale * sqrt(2.))
        return div / sqrt(PI) * exp( -(div*(x - loc)) ** 2.)


cdef class Poisson(Distribution):
    cdef double rvs(self, double loc, double scale) nogil:
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

    cdef double pdf(self, double x, double loc, double scale) nogil:
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
