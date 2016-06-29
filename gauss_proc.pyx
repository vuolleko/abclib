# ****************** Gaussian Process *******************


cdef class GaussProc:
    """
    Implements a Gaussian process for usage with Bayesian optimization.

    See e.g.
    Michael U Gutmann and Jukka Corander: Bayesian optimization for
    likelihood-free inference of simulator-based statistical models,
    arXiv preprint arXiv:1501.03291, 2015.
    """
    cdef double[:, :] params
    cdef double[:] responses
    cdef double[:, :] covariances
    cdef double[:] means
    cdef double[:] diffvec
    cdef double[:, :] chol_K
    cdef double[:] inv_K_diff
    cdef GP_Mean mean_fun
    cdef GP_Cov cov_fun
    cdef double eta_factor
    cdef int n_eval, n_dim, ii_eval, ii_evidence

    def __init__(self, int n_eval, int n_dim, GP_Mean mean_fun, GP_Cov cov_fun):
        """
        A Gaussian process.
        Inputs:
        - n_eval: number of output samples (used for memory allocation)
        - n_dim: number of dimensions
        - mean_fun: instance of GP_Mean (mean function of GP)
        - cov_fun: instance of GP_Cov (covariance function of GP)
        """
        self.n_eval = n_eval
        self.n_dim = n_dim
        self.ii_eval = 0
        self.ii_evidence = 0
        self.params = np.empty((n_eval, n_dim))
        self.responses = np.empty(n_eval)
        self.covariances = np.empty((n_eval, n_eval))
        self.means = np.empty(n_eval)
        self.diffvec = np.empty(n_eval)
        self.mean_fun = mean_fun
        self.cov_fun = cov_fun

    cdef void add_evidence(self, double[:] params_new, double response):
        """
        Add new evidence to GP.
        """
        self.params[self.ii_evidence, :] = params_new
        self.responses[self.ii_evidence] = response
        self.ii_evidence += 1

    cdef void update(self):
        """
        Update GP state.
        """
        cdef int jj

        while self.ii_eval < self.ii_evidence:
            self.means[self.ii_eval] = self.mean_fun.val(self.params[self.ii_eval, :])

            self.covariances[self.ii_eval, self.ii_eval] = self.cov_fun.diag()
            for jj in range(self.ii_eval):
                self.covariances[self.ii_eval, jj] = self.cov_fun.val(
                                self.params[self.ii_eval, :], self.params[jj, :] )
                self.covariances[jj, self.ii_eval] = self.covariances[
                                self.ii_eval, jj ]  # symmetry

            self.ii_eval += 1

    cdef void precalculate_terms(self):
        """
        Evaluate the Cholesky decomposition and precalculate some terms.
        """
        cdef int ii
        for ii in range(self.ii_eval):
            self.diffvec[ii] = self.responses[ii] - self.means[ii]

        self.chol_K = cholesky(self.covariances[:self.ii_eval, :self.ii_eval])
        self.inv_K_diff = solve_ut( self.chol_K.T,
                            solve_lt(self.chol_K, self.diffvec[:self.ii_eval]) )
        self.eta_factor = 2. * log( self.ii_eval**(self.n_dim/2. + 2.) * PI * PI / 0.3 )

    cdef void set_hyperparams(self, double[:] hyperparams):
        """
        Set hyperparameters for the mean and covariance functions.
        """
        # set new hyperparameters using one array
        cdef int offset = self.mean_fun.n_param
        self.mean_fun.set_params( hyperparams[:offset] )
        self.cov_fun.set_params( hyperparams[offset:] )

    cdef double[:] get_hyperparams(self):
        """
        Get hyperparameters for the mean and covariance functions.
        """
        # get current hyperparameters into one array
        cdef int offset = self.mean_fun.n_param
        cdef double[:] hyperparams = np.empty(offset + self.cov_fun.n_param)
        hyperparams[:offset] = self.mean_fun.get_params()
        hyperparams[offset:] = self.cov_fun.get_params()
        return hyperparams

    def neg_log_marginal_lh(self, double[:] hyperparams):
    # cdef double log_marginal_lh(self, double[:] hyperparams):
        """
        -1 * Log marginal likelihood of given hyperparameters.
        -1 to turn minimization into maximization.
        Updates hyperparameters, means and covariances.
        """
        self.set_hyperparams(hyperparams)

        # re-evaluate means, covariances and some terms
        self.ii_eval = 0
        self.update()
        self.precalculate_terms()

        # try to avoid hyperparams resulting in non-pos. definite cov. matrix (SLOW)
        # if not is_pos_def( self.covariances[:self.ii_eval, :self.ii_eval] ):
            # return 1e99

        # evaluate the log marginal likelihood
        cdef int ii
        cdef double lmlh = 0.
        for ii in range(self.ii_eval):
            lmlh += self.diffvec[ii] * self.inv_K_diff[ii]
            lmlh += 2. * log( self.chol_K[ii, ii] )

        return lmlh

    def acquis_fun(self, double[:] params_new):
    # cdef double acquis_fun(self, double[:] params_new):
        """
        Acquisition function (lower confidence bound selection criterion).
        Cox and John, 1992, 1997; Srinivas et al., 2010, 2012.
        """
        cdef int tt = self.ii_eval
        cdef np.ndarray[np.float_t] k_vec = np.empty(tt)
        cdef int ii

        # covariance of new parameters with all of the old ones
        for ii in range(tt):
            k_vec[ii] = self.cov_fun.val(params_new, self.params[ii, :])

        cdef double mean_new = self.mean_fun.val(params_new)
        mean_new += k_vec.T.dot( np.asarray(self.inv_K_diff) )
        cdef double[:] v_vec = solve_lt(self.chol_K, k_vec)
        cdef double var_new = self.cov_fun.diag() - norm2(v_vec)

        return mean_new - sqrt(self.eta_factor * var_new)

    def grad_acquis_fun(self, double[:] params_new):
    # cdef double[:] grad_acquis_fun(self, double[:] params_new):
        """
        Gradient of the acquisition function (lower confidence bound
        selection criterion).
        """
        cdef int nn = self.n_dim
        cdef double[:] grads = np.empty(nn)
        cdef int tt = self.ii_eval
        cdef double[:] k_vec = np.empty(tt)
        cdef np.ndarray[np.float_t] grad_k_vec = np.empty(tt)
        cdef np.ndarray[np.float_t] grad_v_vec = np.empty(tt)
        cdef np.ndarray[np.float_t] v_vec, v_vec_solved
        cdef int ii, jj

        for ii in range(tt):
            k_vec[ii] = self.cov_fun.val(params_new, self.params[ii, :])

        v_vec = np.asarray( solve_lt(self.chol_K, k_vec) )
        cdef double var_new = self.cov_fun.diag() - norm2(v_vec)
        v_vec_solved = np.asarray( solve_ut(self.chol_K.T, v_vec) )

        for jj in range(nn):
            for ii in range(tt):
                grad_k_vec[ii] = self.cov_fun.grad(params_new, self.params[ii, :], jj)
            grad_v_vec = np.asarray( solve_lt(self.chol_K, grad_k_vec) )

            grads[jj] = self.mean_fun.grad(params_new, jj)
            grads[jj] += grad_k_vec.T.dot( np.asarray(self.inv_K_diff) )
            if var_new > 0:  # assume gradient of diagonals = 0
                grads[jj] += sqrt(self.eta_factor / var_new) * grad_v_vec.T.dot(v_vec)

        return np.asarray(grads)

    def acquis_fun_mu(self, double[:] params_new):
        """
        Acquisition function, exploitation only.
        """
        cdef int tt = self.ii_eval
        cdef np.ndarray[np.float_t] k_vec = np.empty(tt)
        cdef int ii

        # covariance of new parameters with all of the old ones
        for ii in range(tt):
            k_vec[ii] = self.cov_fun.val(params_new, self.params[ii, :])

        cdef double mean_new = self.mean_fun.val(params_new)
        mean_new += k_vec.T.dot( np.asarray(self.inv_K_diff) )

        return mean_new

    def grad_acquis_fun_mu(self, double[:] params_new):
        """
        Gradient of the acquisition function, exploitation only.
        """
        cdef int nn = self.n_dim
        cdef double[:] grads = np.empty(nn)
        cdef int tt = self.ii_eval
        cdef np.ndarray[np.float_t] grad_k_vec = np.empty(tt)
        cdef int ii, jj

        for jj in range(nn):
            for ii in range(tt):
                grad_k_vec[ii] = self.cov_fun.grad(params_new, self.params[ii, :], jj)

            grads[jj] = self.mean_fun.grad(params_new, jj)
            grads[jj] += grad_k_vec.T.dot( np.asarray(self.inv_K_diff) )

        return np.asarray(grads)

    def regression(self, double[:, :] param_array):
        """
        Evaluate the GP mean and variance for the input.
        """
        cdef int tt = self.ii_eval
        cdef np.ndarray[np.float_t] k_vec = np.empty(tt)
        cdef int nn = param_array.shape[0]
        cdef double[:] mus = np.empty(nn)
        cdef double[:] sigma2s = np.empty(nn)
        cdef double[:] v_vec
        cdef int ii, jj

        # covariance of parameter array with all known samples
        for jj in range(nn):
            for ii in range(tt):
                k_vec[ii] = self.cov_fun.val(param_array[jj, :], self.params[ii, :])

            mus[jj] = self.mean_fun.val(param_array[jj, :])
            mus[jj] += k_vec.T.dot( np.asarray(self.inv_K_diff) )

            v_vec = solve_lt(self.chol_K, k_vec)
            sigma2s[jj] = self.cov_fun.diag() - norm2(v_vec)

        return np.asarray(mus), np.asarray(sigma2s)

    def exploit(self):
        """
        Find the parameters that minimize the mean function of GP.
        """
        cdef double[:] limits_min, limits_max
        cdef double[:] params_best
        limits_min = np.min(self.params[:self.ii_eval, :], axis=0)
        limits_max = np.max(self.params[:self.ii_eval, :], axis=0)

        params_best = minimize_DIRECT(self.acquis_fun_mu,
                                      limits_min, limits_max)
        return np.asarray(params_best)

# ****************** Mean functions ******************


cdef class GP_Mean:
    """
    A parent class for the mean functions of a Gaussian process.
    Implements the zero-mean case.
    """
    cdef int n_dim, n_param

    def __init__(self, int n_dim, double[:] params_init):
        """
        Inputs:
        - ndim: number of parameters modeled by the GP
        - params_init: a vector of initial parameter values for the mean function
        """
        self.n_dim = n_dim
        self.n_param = 0
        self.set_params(params_init)

    cdef double val(self, double[:] xx):
        """
        Mean function: constant 0.
        """
        return 0.

    cdef double grad(self, double[:] xx, int ind_dim):
        """
        Gradient of mean function: constant 0.
        """
        return 0.

    cdef void set_params(self, double[:] params):
        """
        Set parameters. None here.
        """
        pass

    cdef double[:] get_params(self):
        """
        Return parameters. Empty array here.
        """
        return np.array([])


cdef class GP_Mean_Cvx(GP_Mean):
    """
    Implements a mean function with convex quadratic polynomial mean.
    """
    cdef double[:] a_factor
    cdef double[:] b_factor
    cdef double c_factor

    def __init__(self, int n_dim, double[:] params_init):
        self.a_factor = np.empty(n_dim)
        self.b_factor = np.empty(n_dim)
        super(GP_Mean_Cvx, self).__init__(n_dim, params_init)
        self.n_param = 2 * n_dim + 1

    cdef double val(self, double[:] xx):
        """
        Mean function: sum of convex quadratic polynomials.
        """
        cdef int ii
        cdef double sum0 = 0.

        for ii in range(self.n_dim):
            sum0 += self.a_factor[ii] * xx[ii] * xx[ii]
            sum0 += self.b_factor[ii] * xx[ii] + self.c_factor

        return sum0

    cdef double grad(self, double[:] xx, int ind_dim):
        """
        Gradient of mean function: sum of convex quadratic polynomials.
        """
        return self.a_factor[ind_dim] * xx[ind_dim] + self.b_factor[ind_dim]

    cdef void set_params(self, double[:] params):
        """
        Set parameters in order: a, b, c.
        """
        cdef int ii
        for ii in range(self.n_dim):
            self.a_factor[ii] = params[ii]
            self.b_factor[ii] = params[ii + self.n_dim]
        self.c_factor = params[2 * self.n_dim]

    cdef double[:] get_params(self):
        """
        Return parameters in order: a, b, c.
        """
        cdef int ii
        cdef double[:] params = np.empty(self.n_param)
        for ii in range(self.n_dim):
            params[ii] = self.a_factor[ii]
            params[ii + self.n_dim] = self.b_factor[ii]
        params[2 * self.n_dim] = self.c_factor


# ****************** Covariance functions ******************


cdef class GP_Cov:
    """
    A parent class for the covariance functions of a Gaussian process.
    Implements the constant variance case.
    """
    cdef int n_dim, n_param
    cdef double var_factor

    def __init__(self, int n_dim, double[:] params_init):
        """
        Inputs:
        - ndim: number of parameters modeled by the GP
        - params_init: a vector of initial parameter values for the cov. function
        """
        self.n_dim = n_dim
        self.n_param = 1
        self.set_params(params_init)

    cdef double val(self, double[:] xx, double[:] yy):
        """
        Covariance function: constant.
        """
        return 0.

    cdef double diag(self):
        """
        Variance of parameters.
        """
        return self.var_factor

    cdef double grad(self, double[:] xx, double[:] yy, int ind_dim):
        """
        Gradient of covariance function: constant.
        """
        return 0.

    cdef void set_params(self, double[:] params):
        """
        Set parameters.
        """
        self.var_factor = params[0]

    cdef double[:] get_params(self):
        """
        Set parameters.
        """
        return np.array([ self.var_factor ])


cdef class GP_Cov_Sq_Exp(GP_Cov):
    """
    Implements the squared exponential covariance function for a Gaussian process.
    """
    cdef double sigma2_signal
    cdef double sigma2_obs
    cdef double[:] scale2

    def __init__(self, int n_dim, double[:] params_init):
        self.scale2 = np.empty(n_dim)
        super(GP_Cov_Sq_Exp, self).__init__(n_dim, params_init)
        self.n_param = n_dim + 2

    cdef double val(self, double[:] xx, double[:] yy):
        """
        Covariance function: Squared exponential covariance.
        """
        cdef int nn = xx.shape[0]
        cdef int ii
        cdef double sum0 = 0.
        cdef double temp

        for ii in range(nn):
            temp = (xx[ii] - yy[ii])
            sum0 += temp * temp / self.scale2[ii]

        return self.sigma2_signal * exp(-sum0)

    cdef double diag(self):
        """
        Variance of parameters.
        """
        return self.sigma2_signal + self.sigma2_obs

    cdef double grad(self, double[:] xx, double[:] yy, int ind_dim):
        """
        Gradient of Covariance function: Squared exponential covariance.
        """
        cdef double res = self.val(xx, yy)
        res *= 2. * (yy[ind_dim] - xx[ind_dim]) / self.scale2[ind_dim]
        return res

    cdef void set_params(self, double[:] params):
        """
        Set parameters in order: sigma2_signal, sigma2_obs, scale.
        """
        cdef int ii
        self.sigma2_signal = params[0]
        self.sigma2_obs = params[1]
        for ii in range(self.n_dim):
            self.scale2[ii] = params[ii + 2]

    cdef double[:] get_params(self):
        """
        Return parameters in order: sigma2_signal, sigma2_obs, scale.
        """
        cdef double[:] params = np.empty( self.n_param )
        cdef int ii
        params[0] = self.sigma2_signal
        params[1] = self.sigma2_obs
        for ii in range(self.n_dim):
            params[ii + 2] = self.scale2[ii]
        return params
