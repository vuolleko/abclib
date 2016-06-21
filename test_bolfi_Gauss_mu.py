import numpy as np
import matplotlib.pyplot as plt
import abclib


abclib.init_rand()
true_params = np.array([2.7])
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Mean()]
n_eval = 20
simu = abclib.Simu_Gauss_mu(100, sumstats)
obs = simu(true_params)

limits_min = np.array([-3.])
limits_max = np.array([10.])
hyperp_min = np.array([0., 0., 0.])
hyperp_max = np.array([1., 2., 1.])

n_init = 5
init_x = [limits_min + np.random.rand(1) * (limits_max - limits_min) for ii in range(n_init)]
mean_fun = abclib.GP_Mean(1, np.array([]))
cov_fun = abclib.GP_Cov_Sq_Exp(1, np.array([1., 0.1, 2.]))
gp = abclib.GaussProc(n_eval, 1, mean_fun, cov_fun)


xx, yy = abclib.abc_bolfi(simu, obs, init_x, n_eval, limits_min, limits_max,
                          hyperp_min, hyperp_max, distance, gp,
                          hp_learn_interval=15, sigma_jitter=0.2)

# print np.concatenate((xx, yy[:, np.newaxis]), axis=1)
ind_min = np.argmin(yy)
print "Min evidence {} at {}".format(yy[ind_min], xx[ind_min])
print "Exploited: ", gp.exploit()

x1_array = np.linspace(xx.min(), xx.max(), 50)
mus, sigma2s = gp.regression(x1_array[:, np.newaxis])
sigmas = np.sqrt(sigma2s)

fig, ax = plt.subplots(nrows=2)
ax[0].semilogy(xx[:n_init], yy[:n_init], 'or', label='Initial samples')
ax[0].semilogy(xx[n_init:], yy[n_init:], 'ob', label='From acq. func.')
ax[0].set_ylabel('Discrepancy')
ax[0].set_title('Samples')
ax[0].legend(loc='best', numpoints=1)

ax[1].plot(x1_array, mus)
ax[1].fill_between(x1_array, mus-1.96*sigmas, mus+1.96*sigmas, alpha=0.5)
ax[1].set_xlabel('$\mu$')
ax[1].set_ylabel('Discrepancy')
ax[1].set_title('Posterior mean and 95% confidence interval')
ax[1].set_ylim([0, 10])
plt.show()
