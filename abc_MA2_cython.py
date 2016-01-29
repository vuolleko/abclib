import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
params_true = np.array([1.4, 0.8])
params_guess = np.array([2., 1.])
params_const = np.array([])
n_sets = 1e6
# epsilon = 0.1
epsilon = 4e2
sd = 0.5

simu = abclib.MA2()
observed = simu.run(params_true, params_const, n_obs)
distribs = [abclib.Normal(), abclib.Normal()]
distance = abclib.Distance_L2_Nrlz()
# sumstats = [abclib.Autocov(1), abclib.Autocov(2)]
# distance = abclib.Distance_INT_PER(observed)
sumstats = []
params = abclib.abc_mcmc(simu, params_const, observed, distance, sumstats, distribs, n_sets, epsilon, params_guess, sd)

print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[:, 0]), np.mean(params[:, 1]))

plt.figure()
plt.scatter(params[:, 0], params[:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=10, mew=3)
plt.title('MA(2)')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()
