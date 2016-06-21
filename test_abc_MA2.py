import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
n_samples = 10
params_true = np.array([0.6, 0.2])
n_output = 100000
distance = abclib.Distance_L2()
sumstats = [abclib.SS_Autocov(1), abclib.SS_Autocov(2)]

abclib.init_rand()
simu = abclib.MA2(n_obs, sumstats)
observed = simu(params_true)
print "Range of observed data: {:.3f} .. {:.3f}, mean: {:.3f}".format(np.array(observed).min(), np.array(observed).max(), np.array(observed).mean())

simu = abclib.MA2(n_obs, sumstats, n_samples)
priors = [abclib.Normal(0.454, 1., minval=0.), abclib.Normal(0.014, 1., minval=0.)]
proposals = [abclib.Normal(scale=0.1, minval=0.), abclib.Normal(scale=0.1, minval=0.)]

# ******* ABC-Reject ********
# params, epsilon, distances = abclib.abc_reject(n_output, simu, observed, priors, distance, p_quantile=0.1)

# ******* ABC-MCMC ********
params, epsilon, distances = abclib.abc_reject(1000, simu, observed, priors, distance, p_quantile=0.01)
init_guess = np.mean(params[:, :], axis=0)
print "Trying with initial guess: {}".format( init_guess )
params = abclib.abc_mcmc(n_output, simu, observed, priors, distance, proposals, init_guess, epsilon, symmetric_proposal=True)

# ******* ABC-SEQ-MC ********
# n_seq = 5
# n_output = 10000
# schedule = np.ones(n_seq) * 10.
# params = abclib.abc_seq_mc(n_output, simu, observed, priors, distance, proposals,
#                            schedule, n_seq, p_quantile=0.5)
# params = params[-1, :, :]

# ******* output ************
print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[1000:, 0]), np.mean(params[1000:, 1]))

plt.figure()
plt.scatter(params[1000:, 0], params[1000:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=20, mew=4)
plt.title('MA(2)')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')


fig, ax = plt.subplots(2, sharex=True)
ax[0].plot(params[:, 0])
ax[0].plot([0, n_output], [params_true[0], params_true[0]], 'r', lw=3)
ax[0].set_ylabel('$\\theta_1$')

ax[1].plot(params[:, 1])
ax[1].plot([0, n_output], [params_true[1], params_true[1]], 'r', lw=3)
ax[1].set_ylabel('$\\theta_2$')
ax[1].set_xlabel('Iteration')

plt.show()
