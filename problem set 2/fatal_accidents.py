import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import rv_continuous

data = np.loadtxt('data.txt', skiprows=1)


def likelihood(y, theta):
    """
    Unnormalized Poisson distribution
    """
    res = 1
    for yi in y:
        res *= theta**yi*np.exp(-theta)
    return res

def prior(theta):
    """
    Unnormalized normal distribution
    """
    mu = 30
    sigma = 20
    return np.exp(-(theta - mu)**2/(2*sigma**2))

fatal_accidents = data[:, 1]


def posterior(theta, y):
    """
    Unnormalized posterior distribution
    """
    return likelihood(y, theta)*prior(theta)

theta = np.linspace(0, 70, 10000)
normalization = np.trapz(posterior(theta, fatal_accidents), theta)

def posterior_normalized(theta, y):
    return posterior(theta, y)/normalization

class posterior_normalized_gen(rv_continuous):
    def _pdf(self, theta):
        return posterior_normalized(theta, y=fatal_accidents)

posterior_dist = posterior_normalized_gen(name='posterior_dist')


sns.set()
plt.plot(theta, posterior_normalized(theta, fatal_accidents))
plt.xlabel(r'$\theta$')
plt.ylabel(r'$p(\theta|y)$')
plt.show()
