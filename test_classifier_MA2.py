import numpy as np
import matplotlib.pyplot as plt

import abclib

n_obs = 100
params_true = np.array([1.4, 0.8])
n_output = int(1e5)

abclib.init_rand()
simu = abclib.MA2(n_obs)
observed = simu(params_true)
print "Range of observed data: {:.3f} .. {:.3f}, mean: {:.3f}".format(np.array(observed).min(), np.array(observed).max(), np.array(observed).mean())

priors = [abclib.Uniform(0., 5.), abclib.Uniform(0., 5.)]
proposals = [abclib.Normal(scale=1.), abclib.Normal(scale=1.)]
features = abclib.Feature_triplets(observed)
distance = abclib.Classifier(features)

params, epsilon, distances = abclib.abc_reject(1000, simu, observed, priors, distance, p_quantile=0.1)

params_guess = params.mean(axis=0)

print "Trying with initial guess: {} and epsilon {}".format( params_guess, epsilon )
params = abclib.abc_mcmc(n_output, simu, observed, priors, distance, proposals, params_guess, epsilon, symmetric_proposal=True)

params = params[1000:, :]

print "Posterior means: {:.3f}, {:.3f}".format(np.mean(params[:, 0]), np.mean(params[:, 1]))

plt.figure()
plt.scatter(params[:, 0], params[:, 1])
plt.plot(params_true[0], params_true[1], c='r', marker='x', markersize=10, mew=3)
plt.title('MA(2), Classifier ABC')
plt.xlabel('$\\theta_1$')
plt.ylabel('$\\theta_2$')
plt.show()
