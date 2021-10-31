import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet
from scipy.stats import rv_histogram

alpha = np.array(4*[0.5])

dirichlet = dirichlet(alpha=alpha)

def phi(p):
    alpha = p.sum(axis=2, keepdims=True)
    beta  = p.sum(axis=1, keepdims=True)

    alphabeta = alpha@beta

    return np.sum((p - alphabeta)**2/p, axis=(1,2))

def gamma(p):
    return p[:, 0, 0] + p[:, 1, 1]

def delta(p):
    return p[:, 0, 1]*p[:, 1, 0]/(p[:, 0, 0]*p[:, 1, 1])

N = 5000000
# samples = dirichlet.rvs(size=N)
# samples = samples.reshape((N, 2, 2))

quantiles = [0.05, 0.5, 0.95]

# print('Phi quantiles:', np.quantile(phi(samples), quantiles).round(3))
# print('Gamma quantiles:', np.quantile(gamma(samples), quantiles).round(3))
# print('Delta quantiles:', np.quantile(delta(samples), quantiles).round(3))

# phi_hist = np.histogram(phi(samples), bins=1000000, range = (0, 10000), density=True)
# phi_dist = rv_histogram(phi_hist)
#
# phi_input = np.linspace(0, 50, 100000)
# plt.plot(phi_input, phi_dist.pdf(phi_input))
# plt.show()

n = np.array([24, 27, 34, 26])

dirichlet.alpha = n + 0.5

# samples = dirichlet.rvs(size=N)
# samples = samples.reshape((N, 2, 2))
#
# print('Phi quantiles:', np.quantile(phi(samples), quantiles).round(3))
# print('Gamma quantiles:', np.quantile(gamma(samples), quantiles).round(3))
# print('Delta quantiles:', np.quantile(delta(samples), quantiles).round(3))

mydata = np.array([4, 0, 2, 2])

dirichlet.alpha = n + mydata

samples = dirichlet.rvs(size=N)
samples = samples.reshape((N, 2, 2))

print('Phi quantiles:', np.quantile(phi(samples), quantiles).round(3))
print('Gamma quantiles:', np.quantile(gamma(samples), quantiles).round(3))
print('Delta quantiles:', np.quantile(delta(samples), quantiles).round(3))
