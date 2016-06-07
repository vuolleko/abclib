import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
n_samples = 100
params_true = np.array([1.4, 0.8])
n_output = int(1e5)

abclib.init_rand()
simu = abclib.MA2(n_obs, n_samples)
observed = simu(params_true)
print "Range of observed data: {:.3f} .. {:.3f}, mean: {:.3f}".format(np.array(observed).min(), np.array(observed).max(), np.array(observed).mean())

priors = [abclib.Uniform(0., 5.), abclib.Uniform(0., 5.)]
proposals = [abclib.Normal(scale=1.), abclib.Normal(scale=1.)]
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Autocov(1), abclib.SS_Autocov(2)]
# distance = abclib.Distance_INT_PER(observed)
# distance = abclib.Distance_DTW()
# sumstats = []

# ******* ABC-Reject ********
# params, epsilon, distances = abclib.abc_reject(n_output, simu, observed, priors, distance, sumstats, p_quantile=0.1)

# ******* ABC-MCMC ********
params, epsilon, distances = abclib.abc_reject(10000, simu, observed, priors, distance, sumstats)
init_guess = np.mean(params[:, :], axis=0)
print "Trying with initial guess: {}".format( init_guess )
params = abclib.abc_mcmc(n_output, simu, observed, priors, distance, sumstats, proposals, init_guess, epsilon, symmetric_proposal=True)

# ******* ABC-sample
# params = abclib.sample_abc(simu, observed, distance, sumstats, distribs, n_output, epsilon, n_samples, init_guess[1, :], sd)

# ******* ABC-SEQ-MC ********
# schedule = np.zeros(5)
# params = abclib.abc_seq_mc(n_output, simu, observed, priors, distance, sumstats, schedule, 5, p_quantile=0.5)


print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[1000:, 0]), np.mean(params[1000:, 1]))

plt.figure()
plt.scatter(params[1000:, 0], params[1000:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=10, mew=3)
plt.title('MA(2)')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()