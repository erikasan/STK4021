import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma

@np.vectorize
def likelihood(N, y, p0):
    return gamma(N + 1)/gamma(N - y + 1)*(1 - p0)**N

N  = np.arange(6, 20)
y  = 6
p0 = 0.515

L = likelihood(N, y, p0)

N_max = N[np.argmax(L)]

# sns.set()
# plt.plot(N, L/np.max(L), 'o')
# plt.plot(N_max, 1, 'o', color='red', label=r'Maximum at $N = {}$'.format(N_max))
# plt.xlabel('Number of children of Odin')
# plt.ylabel(r'$\mathcal{L} / \mathcal{L}_{\mathrm{max}}$')
# plt.legend()
# plt.savefig('1a.pdf', type='pdf')
# plt.show()

from scipy.stats import rv_discrete

def posterior_(N, y, p0):
    return likelihood(N, y, p0)/(N + 1)

N = np.arange(6, 51)

normalization = np.sum(posterior_(N, y, p0))

def posterior(N, y, p0):
    return posterior_(N, y, p0)/normalization

# probs = posterior(N, y, p0)
# posterior = rv_discrete(name='posterior', values=(N, probs))
#
# median = posterior.median()
# crd_int = posterior.interval(alpha=0.90)
#
# sns.set()
# plt.plot(N, posterior.pmf(N), 'o', markersize=5)
# plt.plot(median, posterior.pmf(median), 'o', color='red',
#          label=r'Median at $N = {:.0f}$'.format(median))
# plt.xlabel(r'$N$')
# plt.ylabel(r'$P(N|y)$')
# plt.legend()
# plt.savefig('1b.pdf', type='pdf')
# plt.show()
#
# N = np.arange(crd_int[0], crd_int[1])
# print(crd_int)
# print(np.sum(posterior.pmf(N)).round(2))
#
# N = np.arange(crd_int[0], crd_int[1])
# print(np.sum(posterior.pmf(N)).round(2))

def p_myown(N):
    mu = 2
    sigma = 5
    return np.exp(-(N - mu)**2/(2*sigma**2))

def posterior__(N, y, p0):
    return likelihood(N, y, p0)*p_myown(N)

N = np.arange(6, 100)
normalization_ = np.sum(posterior__(N, y, p0))

def posterior_myown(N, y, p0):
    return posterior__(N, y, p0)/normalization_

N = np.linspace(6, 51, 1000)

sns.set()
plt.plot(N, posterior(N, y, p0), label=r'$P(N|y)$')
plt.plot(N, posterior_myown(N, y, p0), label=r'$P_{\mathrm{myown}}( N|y)$')
plt.xlabel(r'$N$')
plt.legend()
plt.savefig('1c.pdf', type='pdf')
plt.show()
