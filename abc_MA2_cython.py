import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
params_true = np.array([1.4, 0.8])
params_guess = np.array([2., 1.])
params_const = np.array([])
n_sets = int(1e4)
# n_samples = 10
sd = np.ones(2) * 1.

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

# ******* ABC-Reject ********
# params, epsilon, distances = abclib.abc_reject(simu, params_const, observed, distance, sumstats, distribs, n_sets, params_guess, sd, p_quantile=0.1)

# ******* ABC-MCMC ********
# params, epsilon, distances = abclib.abc_reject(simu, params_const, observed, distance, sumstats, distribs, n_sets/10, params_guess, sd * 3.)
# params_guess = np.mean(params[:, :], axis=0)

# print "Trying with initial guess: {}".format( params_guess )
# params = abclib.abc_mcmc(simu, params_const, observed, distance, sumstats, distribs, n_sets, epsilon, params_guess, sd)

# ******* ABC-sample
# params = abclib.sample_abc(simu, params_const, observed, distance, sumstats, distribs, n_sets, epsilon, n_samples, params_guess[1, :], sd)

# ******* ABC-SEQ-MC ********
params = abclib.abc_seq_mc(simu, params_const, observed, distance, sumstats, distribs, n_sets, params_guess, sd, 5)


print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[n_sets/2:, 0]), np.mean(params[n_sets/2:, 1]))

plt.figure()
plt.scatter(params[n_sets/2:, 0], params[n_sets/2:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=10, mew=3)
plt.title('MA(2)')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()
