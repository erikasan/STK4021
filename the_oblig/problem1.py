import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import gamma

def likelihood(N, y, p0):
    return gamma(N + 1)/gamma(N - y + 1)*(1 - p0)**N

N  = np.arange(6, 20)
y  = 6
p0 = 0.515

L = likelihood(N, y, p0)

N_max = N[np.argmax(L)]

sns.set()
plt.plot(N, L/np.max(L), 'o')
plt.plot(N_max, 1, 'o', color='red', label=r'Maximum at $N = {}$'.format(N_max))
plt.xlabel('Number of children of Odin')
plt.ylabel(r'$\mathcal{L} / \mathcal{L}_{\mathrm{max}}$')
plt.legend()
plt.savefig('1a.pdf', type='pdf')
plt.show()

# def posterior(N, y, p0):
#     return likelihood(N, y, p0)/(N + 1)
#
# N = np.arange(6, 51)
#
# normalization = np.sum(posterior(N, y,p0))
#
# sns.set()
# plt.plot(N, posterior(N, y, p0)/normalization, 'o', markersize=5)
# plt.xlabel(r'$N$')
# plt.ylabel(r'$P(N|y)$')
# plt.show()
