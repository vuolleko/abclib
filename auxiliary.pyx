# **************** Auxiliary functions ***************

cdef inline double sum_of(double[:] data) nogil:
    """
    Sum of data.
    """
    cdef int ii
    cdef double sum0 = 0.

    for ii in range(data.shape[0]):
        sum0 += data[ii]

    return sum0


cdef inline double weighted_mean_of(double[:] data, double[:] weights) nogil:
    """
    Weighted mean of data.
    """
    cdef int ii
    cdef double sum0 = 0.

    for ii in range(data.shape[0]):
        sum0 += weights[ii] * data[ii]

    return sum0


cdef inline double var_of(double[:] data, double mean) nogil:
    """
    Variance of data.
    """
    cdef int ii
    cdef int nn = data.shape[0]
    cdef double var = 0.
    cdef double temp

    for ii in range(nn):
        temp = data[ii] - mean
        var += temp * temp

    return var / nn


cdef inline double weighted_var_of(double[:] data, double[:] weights) nogil:
    """
    Weighted variance of data. Assumes normalized weights.
    """
    cdef int ii
    cdef double var = 0.
    cdef double temp
    cdef double weighted_mean = weighted_mean_of(data, weights)

    for ii in range(data.shape[0]):
        temp = data[ii] - weighted_mean
        var += weights[ii] * temp * temp

    return var


cdef inline double quantile(double[:] data, double prob):
    """
    Returns the prob-quantile of data. Sorts data in-place!
    """
    sort(data)

    cdef int ii = 1
    prob *= data.shape[0] - 1.

    while ii < prob:
        ii += 1
    ii -= 1

    return data[ii] + (data[ii+1] - data[ii]) * (prob - ii)


cdef inline void sort(double[:] data):
    """
    Top-down merge sort. Sorts the data in ascending order and in-place.
    https://en.wikipedia.org/wiki/Merge_sort
    """
    _tdms_split_merge(data, np.empty_like(data), 0, data.shape[0])


# Split-merge part of merge sort.
cdef inline void _tdms_split_merge(double[:] data, double[:] data2, int ind_start, int ind_end) nogil:

    if (ind_end - ind_start < 2):
        return

    cdef int ind_mid = (ind_start + ind_end) / 2
    _tdms_split_merge(data, data2, ind_start, ind_mid)  # recurse first half
    _tdms_split_merge(data, data2, ind_mid, ind_end)  # recurse second half
    _tdms_merge(data, data2, ind_start, ind_mid, ind_end)  # merge halves

    cdef int ii
    for ii in range(ind_start, ind_end):
        data[ii] = data2[ii]


# Merge part of merge sort.
cdef inline void _tdms_merge(double[:] data, double[:] data2, int ind_start, int ind_mid, int ind_end) nogil:
    cdef int ind0 = ind_start
    cdef int ind1 = ind_mid
    cdef int ii

    # combine via pair-wise comparison
    for ii in range(ind_start, ind_end):
        if ( (ind0 < ind_mid) and ( (ind1 >= ind_end) or (data[ind0] <= data[ind1]) ) ):
            data2[ii] = data[ind0]
            ind0 += 1
        else:
            data2[ii] = data[ind1]
            ind1 += 1