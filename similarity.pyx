# ****************** Distance functions ***************

cdef class Distance(object):
    """
    A dummy parent class for distance functions.
    """
    def __call__(Distance self, double[:] data1, double[:] data2):
        return self.get(data1, data2)

    cdef double get(Distance self, double[:] data1, double[:] data2):
        pass

cdef class Distance_L2(Distance):
    """
    (Squared) L2 distance between two vectors.
    """
    cdef double get(Distance_L2 self, double[:] data1, double[:] data2):
        cdef double result = 0.
        cdef double diff
        cdef int ii

        for ii in range(data1.shape[0]):
            diff = data1[ii] - data2[ii]
            result +=  diff * diff

        return result

cdef class Distance_L1(Distance):
    """
    L1 distance between two vectors.
    """
    cdef double get(Distance_L1 self, double[:] data1, double[:] data2):
        cdef double result = 0.
        cdef int ii

        for ii in range(data1.shape[0]):
            result += (data1[ii] - data2[ii])

        return result

cdef class Distance_L2_Nrlz(Distance):
    """
    (Squared) L2 distance between two vectors normalized by observation.
    """
    cdef double get(Distance_L2_Nrlz self, double[:] data1, double[:] data2):
        cdef double result = 0.
        cdef double temp
        cdef int ii

        for ii in range(data1.shape[0]):
            temp = (data1[ii] - data2[ii]) / data1[ii]
            result += temp * temp

        return result

cdef class Distance_Corr(Distance):
    """
    Distance based on covariance.
    """
    cdef double mean1, stddev1
    def __cinit__(Distance_Corr self, double[:] data):
        self.mean1 = sum_of(data) / data.shape[0]
        self.stddev1 = sqrt( var_of(data) )

    cdef double get(Distance_Corr self, double[:] data1, double[:] data2):
        cdef int nn = data1.shape[0]
        cdef double mean2 = sum_of(data2) / nn
        cdef double stddev2 = sqrt( var_of(data2) )
        cdef double corr = 0.
        cdef double result
        cdef int ii

        for ii in range(nn):
            corr += (data1[ii] - self.mean1) * (data2[ii] - mean2)

        corr /= (self.stddev1 * stddev2)

        result = (1. - corr) / (1. + corr)

        return result

cdef class Distance_INT_PER(Distance):
    """
    Distance based on integrated periodogram.
    """
    cdef int nn
    cdef double[:] window
    cdef double[:] spec0
    def __cinit__(Distance_INT_PER self, double[:] data1):
        self.window = scipy.signal.get_window( ('tukey', 0.1), data1.shape[0])
        self.spec0 = scipy.signal.periodogram(data1, window=self.window)[1]
        self.nn = self.spec0.shape[0]

        cdef int ii
        cdef double spec_sum = sum_of(self.spec0)

        for ii in range(self.nn):
            self.spec0[ii] /= spec_sum

    cdef double get(Distance_INT_PER self, double[:] data1, double[:] data2):
        cdef double[:] spec1 = scipy.signal.periodogram(data2, window=self.window)[1]

        # cdef complex[:] fft = np.fft.rfft(data2)
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

        # cdef double meandif = (sum_of(data1) - sum_of(data2)) / data1.shape[0]

        return dist #+ abs(meandif)


cdef class Distance_DTW(Distance):
    """
    Dynamic time warping without locality constraint.
    """
    cdef int window

    def __cinit__(Distance_DTW self, int window=-1):
        self.window = window

    cdef double get(Distance_DTW self, double[:] data1, double[:] data2):
        cdef int nn = data1.shape[0]
        cdef double[:, :] arr = np.empty((nn, nn))
        cdef int ii, jj
        cdef double eucl
        cdef int jj_min=1, jj_max=nn

        for ii in range(nn):
            for jj in range(nn):
                arr[ii, jj] = INFINITY
        arr[0, 0] = 0.

        for ii in range(1, nn):

            if self.window > 0:
                jj_min = 1 if (ii-self.window < 1) else ii-self.window
                jj_max = nn if (ii+self.window+1 > nn) else ii+self.window+1

            for jj in range(jj_min, jj_max):
                eucl = (data1[ii] - data2[jj])**2.
                arr[ii, jj] = eucl + fmin( arr[ii-1, jj],
                                           fmin( arr[ii, jj-1], arr[ii-1, jj-1] ))

        return arr[nn-1, nn-1]


cdef class Distance_DTW_Keogh(Distance_DTW):
    """
    Dynamic time warping accelerated by Keogh's lower bounds.
    """
    cdef int reach
    cdef double dist_min
    cdef double obs_prev

    def __cinit__(Distance_DTW_Keogh self, int window=-1, int reach=5):
        self.window = window
        self.reach = reach
        self.obs_prev = -1234.1234  # "random"

    cdef double get(Distance_DTW_Keogh self, double[:] data1, double[:] data2):
        cdef double result

        # a rough test to check if the data is "new"
        if self.obs_prev != data1[0]:
            self.obs_prev = data1[0]
            self.dist_min = INFINITY
            # print "set dist to inf"

        # LB_Keogh <= DTW
        result = self.lowerbound_Keogh( data1, data2 )

        # evaluate DTW only if LB_Keogh small enough
        if self.dist_min > result:
            result = super(Distance_DTW_Keogh, self).__call__(data1, data2)
            self.dist_min = result

        return result

    cdef inline double lowerbound_Keogh(Distance_DTW_Keogh self, double[:] data1, double[:] data2) nogil:
        """
        Keogh's lower bound for dynamic time warping.
        """
        cdef double lb_sum = 0.
        cdef double lowerbound, upperbound
        cdef int nn = data1.shape[0]
        cdef int ii, jj, jj_min, jj_max

        for ii in range(nn):
            lowerbound = INFINITY
            upperbound = -INFINITY
            jj_min = 0 if (ii-self.reach < 0) else ii-self.reach
            jj_max = nn if (ii+self.reach+1 > nn) else ii+self.reach+1
            for jj in range(jj_min, jj_max):
                lowerbound = fmin( data2[jj], lowerbound )
                upperbound = fmax( data2[jj], upperbound )

            if data1[ii] > upperbound:
                lb_sum += (data1[ii] - upperbound)**2.
            elif data1[ii] < lowerbound:
                lb_sum += (data1[ii] - lowerbound)**2.

        return lb_sum


# ****************** Summary statistics ***************


cdef class SummaryStat:
    """
    A dummy parent class for summary statistics functions.
    """
    def __call__(self, double[:] data):
        return self.get(data)

    cdef double get(SummaryStat self, double[:] data):
        pass

cdef class SS_Autocov(SummaryStat):
    """
    Autocovariance of a vector (sort of) with lag.
    """
    cdef int lag

    def __cinit__(SS_Autocov self, int lag):
        self.lag = lag

    cdef double get(SS_Autocov self, double[:] data):
        cdef double result = 0.
        cdef int ii

        for ii in range(self.lag+1, data.shape[0]):
            result += data[ii] * data[ii-self.lag]

        return result

cdef class SS_Mean(SummaryStat):
    """
    Arithmetic mean of vector.
    """
    cdef double get(SS_Mean self, double[:] data):
        return sum_of(data) / data.shape[0]

cdef class SS_Var(SummaryStat):
    """
    Variance of a vector.
    """
    cdef double get(SS_Var self, double[:] data):
        return var_of(data)

cdef class SS_MeanRatio2(SummaryStat):
    """
    Mean of x_i / x_{i-1}
    """
    cdef double get(SS_MeanRatio2 self, double[:] data):
        cdef int nn = data.shape[0]
        cdef double[:] ratio = np.empty(nn-1)
        cdef int ii

        for ii in range(1, nn):
            if data[ii-1] > 0.:
                ratio[ii] = data[ii] / data[ii-1]

        return sum_of(ratio) / (nn-1)

cdef class SS_MedianRatio2(SummaryStat):
    """
    Median of x_i / x_{i-1}
    """
    cdef double get(SS_MedianRatio2 self, double[:] data):
        cdef int nn = data.shape[0]
        cdef double[:] ratio = np.empty(nn-1)
        cdef int ii

        for ii in range(1, nn):
            if data[ii-1] > 0.:
                ratio[ii] = data[ii] / data[ii-1]
            else:
                ratio[ii] = data[ii] * 1e99

        return np.median(ratio)


# ****************** Normalizing function ***************

@cython.profile(False)
cdef inline void normalize(double[:] data) nogil:
    """
    Normalize vector inplace.
    D -> (D - mean) / variance
    """
    cdef int nn = data.shape[0]
    cdef double mean = sum_of(data) / nn
    cdef double var = var_of(data)
    cdef int ii

    for ii in range(nn):
        data[ii] -= mean
        data[ii] /= var


# ****************** Helper functions ***************

cdef inline double sum_of(double[:] data) nogil:
    cdef int ii
    cdef double sum0 = 0.

    for ii in range(data.shape[0]):
        sum0 += data[ii]

    return sum0


cdef inline double var_of(double[:] data) nogil:
    cdef int ii
    cdef int nn = data.shape[0]
    cdef double var = 0.
    cdef double mean = sum_of(data) / nn
    cdef double temp

    for ii in range(nn):
        temp = data[ii] - mean
        var += temp * temp

    return var / nn
