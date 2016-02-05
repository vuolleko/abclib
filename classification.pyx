# **************** Classification-based discrepancy ***************
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# For convenience, this inherits the Distance class.
class Classifier(Distance):
    """
    Approximates distance based on accuracy of classification.
    See "Statistical inference of intractable generative models via classification",
    Gutmann et al., arXiv:1407.4981, 2015.
    """
    def __init__(self, int n_observed, int n_folds=5):
        self.n_folds = n_folds
        self.nn = 2 * n_observed
        cdef int ii

        # create a vector of labels, 0=observed, 1=simulated
        self.labels = np.empty(self.nn, dtype=np.int32)
        for ii in range(n_observed):
            self.labels[ii] = 0
            self.labels[ii+n_observed] = 1

        # create a vector of indices 0:n_folds
        self.inds = np.empty(self.nn, dtype=np.int32)
        cdef int jj = -1
        fold_point = self.nn / self.n_folds
        for ii in range(self.nn):
            if (ii % fold_point == 0):
                jj += 1
            self.inds[ii] = jj


    def __call__(self, double[:] observed, double[:] simulated):
        accuracy = 0.
        cdef int ii, jj
        cdef int nn1 = observed.shape[0]

        # concatenate observed and simulated data
        data = np.concatenate((observed, simulated)).reshape(-1, 1)

        # randomize order of indices
        np.random.shuffle(self.inds)

        for ii in range(self.n_folds):
            qda = QuadraticDiscriminantAnalysis()
            selected = self.inds == ii
            qda.fit(data[selected], self.labels[selected])
            accuracy += qda.score(data[np.invert(selected)], self.labels[np.invert(selected)])

        return accuracy / self.n_folds


# ****************** Classifiers ***************

cdef int[:] nearest_neighbor(double[:,:] train, int[:] labels, double[:,:] test, Distance distance):
    """
    1-Nearest-Neighbor classification. Accelerated with LB_Keogh lower bounds.
    """
    cdef int n_train = train.shape[0]
    cdef int n_test = test.shape[0]
    cdef int[:] predicted = np.empty(n_test, dtype=np.int32)
    cdef int ii, jj
    cdef double dist_min

    for ii in range(n_test):
        dist_min = INFINITY
        for jj in range(n_train):
            dist = distance( test[ii, :], train[jj, :] )
            if dist < dist_min:
                dist_min = dist
                predicted[ii] = labels[jj]

    return predicted


