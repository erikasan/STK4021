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

N = 500000
samples = dirichlet.rvs(size=N)
samples = samples.reshape((N, 2, 2))

phi_hist = np.histogram(phi(samples), bins=1000, range = (0, 1), density=True)
phi_dist = rv_histogram(phi_hist)

phi_input = np.linspace(0, 0.5, 100000)
plt.plot(phi_input, phi_dist.pdf(phi_input))
plt.show()
