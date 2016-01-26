# ****************** Distance functions ***************

cdef class Distance:
    """
    A dummy parent class for distance functions.
    """
    def __call__(self, double[:] observed, double[:] simulated):
        pass


cdef class Distance_L2(Distance):
    """
    (Squared) L2 distance between two vectors.
    """
    def __call__(self, double[:] observed, double[:] simulated):
        cdef double result = 0.
        cdef unsigned int ii

        for ii in range(observed.shape[0]):
            result += (observed[ii] - simulated[ii]) ** 2.

        return result

cdef class Distance_L1(Distance):
    """
    L1 distance between two vectors.
    """
    def __call__(self, double[:] observed, double[:] simulated):
        cdef double result = 0.
        cdef unsigned int ii

        for ii in range(observed.shape[0]):
            result += (observed[ii] - simulated[ii])

        return result


# ****************** Summary statistics ***************

cdef class SummaryStat:
    """
    A dummy parent class for summary statistics functions.
    """
    def __call__(self, double[:] data):
        pass

cdef class Autocov(SummaryStat):
    """
    Autocovariance of a vector (sort of) with lag.
    """
    cdef unsigned int lag

    def __init__(self, unsigned int lag):
        self.lag = lag

    def __call__(self, double[:] data):
        cdef double result = 0.
        cdef unsigned int ii

        for ii in range(self.lag+1, data.shape[0]):
            result += data[ii] * data[ii-self.lag]

        return result


@cython.profile(False)
cdef inline void normalize(double[:] vector) nogil:
    cdef double mean = 0.
    cdef double var = 0.
    cdef unsigned int ii

    for ii in range(vector.shape[0]):
        mean += vector[ii]

    mean /= vector.shape[0]

    for ii in range(vector.shape[0]):
        vector[ii] -= mean
        var += vector[ii] * vector[ii]

    var /= vector.shape[0]

    for ii in range(vector.shape[0]):
        vector[ii] /= var
