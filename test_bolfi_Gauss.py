import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import abclib

abclib.init_rand()
true_params = np.array([1.3, 2.5])
simu = abclib.Simu_Gauss(100)
obs = simu(true_params)
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Mean(), abclib.SS_Var()]
n_eval = 100
n_init = 10

limits_min = np.array([0., 0.])
limits_max = np.array([4., 4.])
hyperp_min = np.array([0., 0., 0., 0.])
hyperp_max = np.array([100., 100., 200., 100.])

init_x = [limits_min + np.random.rand(2) * (limits_max - limits_min) for ii in range(n_init)]

mean_fun = abclib.GP_Mean(2, np.array([]))
cov_fun = abclib.GP_Cov_Sq_Exp(2, np.array([1.1, 1.2, 4.3, 4.3]))
gp = abclib.GaussProc(n_eval, 2, mean_fun, cov_fun)


xx, yy = abclib.abc_bolfi(simu, obs, init_x, n_eval, limits_min, limits_max,
                          hyperp_min, hyperp_max, distance, sumstats, gp,
                          hp_learn_interval=20, sigma_jitter=0.2)

print np.concatenate((xx, yy[:, np.newaxis]), axis=1)
ind_min = np.argmin(yy)
print "Min evidence {} at {}".format(yy[ind_min], xx[ind_min])
print "Exploited: ", gp.exploit()

x1_array = np.linspace(limits_min[0], limits_max[0], 30)
x2_array = np.linspace(limits_min[1], limits_max[1], 30)
x1m, x2m = np.meshgrid(x1_array, x2_array)
# x2_array = np.ones(50) * true_params[1]
x1x2_array = np.concatenate((x1m.flatten()[:, np.newaxis], x2m.flatten()[:, np.newaxis]), axis=1)
mus, sigmas = gp.regression(x1x2_array)
mus = mus.reshape(x1m.shape)
mus = np.where( mus > 0, np.log10(mus), 0.)
sigmas = sigmas.reshape(x1m.shape)

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.semilogy(xx[:n_init], yy[:n_init], 'or', label='Initial samples')
ax1.semilogy(xx[n_init:], yy[n_init:], 'ob', label='Based on acquisition function')
ax1.set_title('Samples')
ax1.set_ylabel('Distance')

ax2 = fig.add_subplot(312, projection='3d')
ax2.plot_surface(x1m, x2m, mus, alpha=0.3, rstride=3, cstride=3)
xlim = [-2, x1_array[-1]]
ylim = [x2_array[0], 6]
# zlim = [-1, mus.max()]
ax2.contour(x1m, x2m, mus, zdir='x', offset=xlim[0])
ax2.contour(x1m, x2m, mus, zdir='y', offset=ylim[1])
# ax2.contour(x1m, x2m, mus, zdir='z', offset=zlim[0])
ax2.set_xlim(xlim)
ax2.set_ylim(ylim)
# ax2.set_zlim(zlim)
ax2.set_xlabel('$\mu$')
ax2.set_ylabel('$\sigma$')
ax2.set_title('Posterior mean')

ax3 = fig.add_subplot(313, projection='3d')
ax3.plot_surface(x1m, x2m, sigmas, alpha=0.3, rstride=3, cstride=3)
xlim = [-2, x1_array[-1]]
ylim = [x2_array[0], 6]
ax3.contour(x1m, x2m, sigmas, zdir='x', offset=xlim[0])
ax3.contour(x1m, x2m, sigmas, zdir='y', offset=ylim[1])
ax3.set_xlim(xlim)
ax3.set_ylim(ylim)
ax3.set_xlabel('$\mu$')
ax3.set_ylabel('$\sigma$')
ax3.set_title('Posterior variance')

if np.any( np.isnan(mus + sigmas) ):
    print "Bad luck! Non-pos. def. covariance matrix..."
# print mus
# print sigmas

plt.show()
