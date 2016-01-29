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

cdef class Distance_L2_Nrlz(Distance):
    """
    (Squared) L2 distance between two vectors normalized by observation.
    """
    def __call__(self, double[:] observed, double[:] simulated):
        cdef double result = 0.
        cdef unsigned int ii

        for ii in range(observed.shape[0]):
            result += ((observed[ii] - simulated[ii]) / observed[ii]) ** 2.

        return result

# cdef class Distance_W_DLS(Distance):
cdef class Distance_INT_PER(Distance):
    """
    Distance based on integrated periodogram.
    """
    cdef int nn
    cdef double[:] window
    cdef double[:] spec0
    def __init__(self, double[:] observed):
        self.window = scipy.signal.get_window( ('tukey', 0.1), observed.shape[0])
        self.spec0 = scipy.signal.periodogram(observed, window=self.window)[1]
        self.nn = self.spec0.shape[0]

        cdef int ii
        cdef double spec_sum = sum_of(self.spec0)

        for ii in range(self.nn):
            self.spec0[ii] /= spec_sum

    def __call__(self, double[:] observed, double[:] simulated):
        cdef double[:] spec1 = scipy.signal.periodogram(simulated, window=self.window)[1]

        # cdef complex[:] fft = np.fft.rfft(simulated)
        # cdef double[:] spec1 = np.empty(nn)

        cdef int ii
        cdef double spec_sum = sum_of(spec1)

        for ii in range(self.nn):
            spec1[ii] /= spec_sum

        cdef double cumsum = 0.
        cdef double dist = 0.
        for ii in range(self.nn):
            cumsum += self.spec0[ii] - spec1[ii]
            dist += abs(cumsum)

        # cdef double meandif = (sum_of(observed) - sum_of(simulated)) / observed.shape[0]

        return dist #+ abs(meandif)

# ****************** Summary statistics ***************

cdef class SummaryStat:
    """
    A dummy parent class for summary statistics functions.
    """
    def __call__(self, double[:] data):
        pass

cdef class SS_Autocov(SummaryStat):
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

cdef class SS_Mean(SummaryStat):
    """
    Arithmetic mean of vector.
    """
    def __call__(self, double[:] data):
        return sum_of(data) / data.shape[0]


# ****************** Normalizing function ***************

@cython.profile(False)
cdef inline void normalize(double[:] vector) nogil:
    """
    Normalize vector inplace.
    D -> (D - mean) / variance
    """
    cdef double mean = 0.
    cdef double var = 0.
    cdef unsigned int ii

    cdef unsigned int nn = vector.shape[0]

    for ii in range(nn):
        mean += vector[ii]

    mean /= nn

    for ii in range(nn):
        vector[ii] -= mean
        var += vector[ii] * vector[ii]

    var /= nn
    # var /= (nn-1)

    for ii in range(nn):
        vector[ii] /= var


# ****************** Helper functions ***************

cpdef sum_of(double[:] data):
    cdef unsigned int ii
    cdef double sum0 = 0.

    for ii in range(data.shape[0]):
        sum0 += data[ii]

    return sum0
