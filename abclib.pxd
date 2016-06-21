cdef class Distance(object):
    """
    A parent class for distance functions.
    """
    cdef double get(Distance self, double[:] data1, double[:] data2)

cdef class SummaryStat:
    """
    A parent class for summary statistics functions.
    """
    cdef double get(SummaryStat self, double[:] data)

cdef class Simulator:
    """
    A parent class for simulators.
    """
    cdef int n_simu, n_samples, normalized, n_sumstats
    cdef list sumstats
    cdef double[:] run(self, double[:] params)
    cdef double[:] run1(self, double[:] params)

# cdef class Features(object):
#     cdef int n_features
#     cdef int nn
#     cdef double[:,:] data_features
#     cdef void set(Features self, double[:] simulated) nogil
#     cdef double[:,:] get_view(Features self)
