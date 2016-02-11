# **************** Classification-based discrepancy ***************
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# For convenience, this inherits the Distance class.
cdef class Classifier(Distance):
    """
    Approximates distance based on accuracy of classification.
    See "Statistical inference of intractable generative models via classification",
    Gutmann et al., arXiv:1407.4981, 2015.
    """
    cdef int n_folds, nn, nn_1fold, n_features
    cdef Distance distance
    cdef int[:] labels
    cdef int[:] inds

    def __init__(self, int n_observed, int n_folds=5, int n_features=1,
                 Distance distance=Distance_L2()):
        self.n_folds = n_folds
        self.n_features = n_features
        self.nn = 2 * n_observed
        self.distance = distance
        cdef int ii

        # create a vector of labels, 0=observed, 1=simulated
        self.labels = np.empty(self.nn, dtype=np.int32)
        for ii in range(n_observed):
            self.labels[ii] = 0
            self.labels[ii+n_observed] = 1

        # create a vector of indices 0:n_folds
        self.inds = np.empty(self.nn, dtype=np.int32)
        cdef int jj = -1
        self.nn_1fold = self.nn / self.n_folds
        for ii in range(self.nn):
            if (ii % self.nn_1fold == 0):
                jj += 1
            self.inds[ii] = jj


    cdef double get(Classifier self, double[:] data1, double[:] data2):
        cdef int matches = 0
        cdef int ii, jj, ind_test, ind_train
        cdef int nn1 = data1.shape[0]
        cdef double[:,:] train_data = np.empty((self.nn_1fold, self.n_features))
        cdef int[:] train_labels = np.empty(self.nn_1fold, dtype=np.int32)
        cdef double[:,:] test_data = np.empty((self.nn - self.nn_1fold, self.n_features))
        cdef int[:] test_labels = np.empty(self.nn - self.nn_1fold, dtype=np.int32)

        # concatenate observed and simulated data
        cdef data = np.empty((self.nn, self.n_features))
        for ii in range(nn1):
            data[ii, 0] = data1[ii]
            data[ii+nn1, 0] = data2[ii]

        # data = np.concatenate((observed, simulated)).reshape(-1, 1)

        # randomize order of indices
        np.random.shuffle(self.inds)

        for ii in range(self.n_folds):
            ind_train = 0
            ind_test = 0
            for jj in range(self.nn):
                if (self.inds[jj] == ii):
                    train_data[ind_train, 0] = data[jj, 0]
                    train_labels[ind_train] = self.labels[jj]
                    ind_train += 1
                else:
                    test_data[ind_test, 0] = data[jj, 0]
                    test_labels[ind_test] = self.labels[jj]
                    ind_test += 1

            # selected = self.inds == ii
            # qda = QuadraticDiscriminantAnalysis()
            # qda.fit(data[selected], self.labels[selected])
            # accuracy += qda.score(data[np.invert(selected)], self.labels[np.invert(selected)])
            # datasel = data[selected]
            # labelsel = self.labels[selected]
            # datasel2 = data[np.invert(selected)]
            predicted = nearest_neighbor(train_data, train_labels, test_data, self.distance)
            # predicted = nearest_neighbor(data[selected], self.labels[selected],
            #                              data[ np.invert(selected) ], self.distance)
            # modelsols = self.labels[ np.invert(selected) ]
            for jj in range(test_data.shape[0]):
                if predicted[jj] == test_labels[jj]:
                    matches += 1
            # matches += np.sum(predicted == self.labels[ np.invert(selected) ])

        return 1. * matches / ( (self.n_folds - 1) * self.nn )


# ****************** Classifiers ***************


cdef int[:] nearest_neighbor(double[:,:] train, int[:] labels, double[:,:] test, Distance distance):
    """
    1-Nearest-Neighbor classification.
    """
    cdef int n_train = train.shape[0]
    cdef int n_test = test.shape[0]
    cdef int n_features = train.shape[1]
    cdef int[:] predicted = np.empty(n_test, dtype=np.int32)
    cdef int ii, jj, kk
    cdef double dist_min, dist
    cdef double[:] test1 = np.empty(n_features)
    cdef double[:] train1 = np.empty(n_features)

    for ii in range(n_test):
        dist_min = INFINITY
        for jj in range(n_train):
            for kk in range(n_features):
                test1[kk] = test[ii, kk]
                train1[kk] = train[ii, kk]

            # dist = test1[0]*test1[0] - train1[0]*train1[0]
            dist = distance.get( test1, train1 )
            if dist < dist_min:
                dist_min = dist
                predicted[ii] = labels[jj]

    return predicted


