# **************** Classification-based discrepancy ***************
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# For convenience, this inherits the Distance class.
cdef class Classifier(Distance):
    """
    Approximates distance based on accuracy of classification.
    See "Statistical inference of intractable generative models via classification",
    Gutmann et al., arXiv:1407.4981, 2015.
    """
    cdef int n_folds, nn, nn1, nn_1fold, n_features
    cdef Distance distance
    cdef Features features
    cdef int[:] labels
    cdef int[:] inds
    cdef double[:,:] train_data
    cdef int[:] train_labels
    cdef double[:,:] test_data
    cdef int[:] test_labels
    cdef int[:] predicted
    cdef double[:,:] data_features

    def __cinit__(Classifier self, Features features,
                  int n_folds=5, Distance distance=Distance_L2()):

        self.features = features
        self.data_features = self.features.get_view()
        self.n_features = self.data_features.shape[1]
        self.nn = self.data_features.shape[0]
        self.nn1 = self.nn / 2
        self.n_folds = n_folds
        self.distance = distance
        cdef int ii

        # create a vector of labels, 0=observed, 1=simulated
        self.labels = np.empty(self.nn, dtype=np.int32)
        for ii in range(self.nn1):
            self.labels[ii] = 0
            self.labels[ii+self.nn1] = 1

        # create a vector of indices 0:n_folds
        self.inds = np.empty(self.nn, dtype=np.int32)
        cdef int jj = -1
        self.nn_1fold = self.nn / self.n_folds
        for ii in range(self.nn):
            if (ii % self.nn_1fold == 0):
                jj += 1
            self.inds[ii] = jj

        self.train_data = np.empty((self.nn_1fold, self.n_features))
        self.train_labels = np.empty(self.nn_1fold, dtype=np.int32)
        self.test_data = np.empty((self.nn - self.nn_1fold, self.n_features))
        self.test_labels = np.empty(self.nn - self.nn_1fold, dtype=np.int32)
        self.predicted = np.empty(self.nn - self.nn_1fold, dtype=np.int32)



    cdef double get(Classifier self, double[:] data1, double[:] data2):
        cdef int matches = 0
        cdef int ii, jj, kk, ind_test, ind_train

        # add simulated data features
        self.features.set(data2)

        # randomize order of indices
        np.random.shuffle(self.inds)

        for ii in range(self.n_folds):
            ind_train = 0
            ind_test = 0
            for jj in range(self.nn):
                if (self.inds[jj] == ii):
                    for kk in range(self.n_features):
                        self.train_data[ind_train, kk] = self.data_features[jj, kk]
                    self.train_labels[ind_train] = self.labels[jj]
                    ind_train += 1
                else:
                    for kk in range(self.n_features):
                        self.test_data[ind_test, kk] = self.data_features[jj, kk]
                    self.test_labels[ind_test] = self.labels[jj]
                    ind_test += 1

            # selected = self.inds == ii
            # qda = QuadraticDiscriminantAnalysis()
            # qda.fit(data[selected], self.labels[selected])
            # accuracy += qda.score(data[np.invert(selected)], self.labels[np.invert(selected)])
            nearest_neighbor(self.train_data, self.train_labels, self.test_data,
                             self.predicted, self.distance)

            for jj in range(self.test_data.shape[0]):
                if self.predicted[jj] == self.test_labels[jj]:
                    matches += 1

        return 1. * matches / ( (self.n_folds - 1) * self.nn )


# ****************** Classifiers ***************


cdef nearest_neighbor(double[:,:] train, int[:] labels, double[:,:] test, int[:] predicted, Distance distance):
    """
    1-Nearest-Neighbor classification.
    """
    cdef int n_train = train.shape[0]
    cdef int n_test = test.shape[0]
    cdef int n_features = train.shape[1]
    cdef int ii, jj, kk
    cdef double dist_min, dist
    cdef double[:] test1
    cdef double[:] train1

    if n_features > 1:
        test1 = np.empty(n_features)
        train1 = np.empty(n_features)

    for ii in range(n_test):
        dist_min = INFINITY
        for jj in range(n_train):

            if n_features == 1:  # use L2
                dist = test[ii, 0]*test[ii, 0] - train[ii, 0]*train[ii, 0]

            else:
                for kk in range(n_features):
                    test1[kk] = test[ii, kk]
                    train1[kk] = train[ii, kk]

                dist = distance.get( test1, train1 )

            if dist < dist_min:
                dist_min = dist
                predicted[ii] = labels[jj]


# ****************** Features ***************


cdef class Features(object):
    """
    A dummy parent class for creating feature matrices y from data vectors x.
    """
    cdef int n_features
    cdef int nn
    cdef double[:,:] data_features

    def __cinit__(Features self, double[:] observed):
        pass

    def __call__(Features self, double[:] simulated):
        self.get(simulated)

    cdef void set(Features self, double[:] data) nogil:
        pass

    cdef double[:,:] get_view(Features self):
        return self.data_features


cdef class Feature_pairs(Features):
    """
    y_i = ( x_i, x_{i+1} )
    """
    def __cinit__(Feature_pairs self, double[:] observed):
        self.n_features = 2
        self.nn = observed.shape[0] - 1  # no pair for the last item
        self.data_features = np.empty((self.nn * 2, self.n_features))
        cdef int ii

        # assign observed data to the beginning of the feature matrix
        for ii in range(self.nn):
            self.data_features[ii, 0] = observed[ii]
            self.data_features[ii, 1] = observed[ii+1]

    cdef void set(Feature_pairs self, double[:] simulated) nogil:
        cdef int ii

        # assign simulated data to the end of the feature matrix
        for ii in range(self.nn):
            self.data_features[ii+self.nn, 0] = simulated[ii]
            self.data_features[ii+self.nn, 1] = simulated[ii+1]

