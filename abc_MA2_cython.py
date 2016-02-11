import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
params_true = np.array([1.4, 0.8])
params_guess = np.array([2., 1.])
params_const = np.array([])
n_sets = 1e6
epsilon = 50.
# n_samples = 10
sd = 0.1

abclib.init_rand()
simu = abclib.MA2()
observed = simu(params_true, params_const, n_obs)
print "Range of observed data: {:.3f} .. {:.3f}, mean: {:.3f}".format(np.array(observed).min(), np.array(observed).max(), np.array(observed).mean())

distribs = [abclib.Normal(), abclib.Normal()]
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Autocov(1), abclib.SS_Autocov(2)]
# distance = abclib.Distance_INT_PER(observed)
# distance = abclib.Distance_DTW()
# sumstats = []

params_guess = abclib.abc_reject(simu, params_const, observed, distance, sumstats, distribs, 2, epsilon, params_guess, sd * 3.)
params_guess = np.array(params_guess[1, :])

print "Trying with initial guess: {}".format( params_guess )
params = abclib.abc_mcmc(simu, params_const, observed, distance, sumstats, distribs, n_sets, epsilon, params_guess, sd)

# params = abclib.sample_abc(simu, params_const, observed, distance, sumstats, distribs, n_sets, epsilon, n_samples, params_guess[1, :], sd)

print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[:, 0]), np.mean(params[:, 1]))

plt.figure()
plt.scatter(params[:, 0], params[:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=10, mew=3)
plt.title('MA(2)')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()
