import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import abclib

abclib.init_rand()
params_true = np.array([2.3, 1.5])
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Mean(), abclib.SS_Var()]
n_eval = 100
n_init = 10
simu = abclib.Simu_Gauss(100, sumstats)
obs = simu(params_true)

limits_min = np.array([-2., 0.])
limits_max = np.array([5., 5.])
hyperp_min = np.array([0., 0., 0., 0.])
hyperp_max = np.array([10., 10., 5., 5.])

init_x = [limits_min + np.random.rand(2) * (limits_max - limits_min) for ii in range(n_init)]

mean_fun = abclib.GP_Mean(2, np.array([]))
cov_fun = abclib.GP_Cov_Sq_Exp(2, np.array([1., 1., 1., 1.]))
gp = abclib.GaussProc(n_eval, 2, mean_fun, cov_fun)


xx, yy = abclib.abc_bolfi(simu, obs, init_x, n_eval, limits_min, limits_max,
                          hyperp_min, hyperp_max, distance, gp,
                          hp_learn_interval=999, sigma_jitter=0.2)

print np.concatenate((xx, yy[:, np.newaxis]), axis=1)
ind_min = np.argmin(yy)
print "Min evidence {} at {}".format(yy[ind_min], xx[ind_min])
print "Exploited: ", gp.exploit()

x1_array = np.linspace(limits_min[0], limits_max[0], 30)
x2_array = np.linspace(limits_min[1], limits_max[1], 30)
x1m, x2m = np.meshgrid(x1_array, x2_array)
x1x2_array = np.concatenate((x1m.flatten()[:, np.newaxis], x2m.flatten()[:, np.newaxis]), axis=1)
mus, sigmas = gp.regression(x1x2_array)
mus = mus.reshape(x1m.shape)
mus = np.where( mus > 0, np.log10(mus), 0.)
sigmas = sigmas.reshape(x1m.shape)

fig = plt.figure()
ax0 = fig.add_subplot(111)
ax0.plot(xx[:n_init, 0], xx[:n_init, 1], 'or', label='Initial samples')
ax0.plot(xx[n_init:, 0], xx[n_init:, 1], 'ob', label='Based on acquisition function')
ax0.plot(params_true[0], params_true[1], c='r', marker='x', linestyle='none', markersize=20, mew=4, label='True parameters')
ax0.set_xlabel('$\\mu$')
ax0.set_ylabel('$\\sigma^2$')
ax0.legend(loc='best', numpoints=1, framealpha=0.5)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.semilogy(xx[:n_init, 0], yy[:n_init], 'or', label='Initial samples $\mu$')
ax1.semilogy(xx[n_init:, 0], yy[n_init:], 'ob', label='Based on acquisition function $\mu$')
ax1.semilogy(xx[:n_init, 1], yy[:n_init], 'xr', label='Initial samples $\sigma^2$')
ax1.semilogy(xx[n_init:, 1], yy[n_init:], 'xb', label='Based on acquisition function $\sigma^2$')
ax1.set_title('Samples')
ax1.set_ylabel('Distance')

fig = plt.figure()
ax2 = fig.add_subplot(211, projection='3d')
ax2.plot_surface(x1m, x2m, mus, alpha=0.3, rstride=3, cstride=3)
xlim = [-3, x1_array[-1]]
ylim = [x2_array[0], 7]
zlim = [mus.min()-1, mus.max()]
ax2.contour(x1m, x2m, mus, zdir='x', offset=xlim[0])
ax2.contour(x1m, x2m, mus, zdir='y', offset=ylim[1])
ax2.contour(x1m, x2m, mus, zdir='z', offset=zlim[0])
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
ax2.set_zlim(zlim)
ax2.set_xlabel('$\mu$')
ax2.set_ylabel('$\sigma^2$')
ax2.set_title('Posterior mean')

ax3 = fig.add_subplot(212, projection='3d')
ax3.plot_surface(x1m, x2m, sigmas, alpha=0.3, rstride=3, cstride=3)
zlim = [sigmas.min()-1, sigmas.max()]
ax3.contour(x1m, x2m, sigmas, zdir='x', offset=xlim[0])
ax3.contour(x1m, x2m, sigmas, zdir='y', offset=ylim[1])
ax3.contour(x1m, x2m, sigmas, zdir='z', offset=zlim[0])
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_zlim(zlim)
ax3.set_xlabel('$\mu$')
ax3.set_ylabel('$\sigma^2$')
ax3.set_title('Posterior variance')

plt.show()
