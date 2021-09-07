import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def betadist(theta, alpha, beta):
    return theta**(alpha - 1)*(1 - theta)**(beta - 1)

def model(y, theta, n):
    return betadist(theta, alpha=y+1, beta=n-y+1)

def unidist(theta):
    return betadist(theta, 1, 1)

def beta28(theta):
    return betadist(theta, 2, 8)

def beta82(theta):
    return betadist(theta, 8, 2)

def evenmix(theta):
    return beta82(theta) + beta28(theta)


n = 50

@np.vectorize
def Q(y, prior):
    theta1 = np.linspace(0, 0.15, 150)
    theta2 = np.linspace(0.15, 1, 1000 - 150)
    numer = np.trapz(model(y, theta1, n)*prior(theta1), theta1)
    denom = np.trapz(model(y, theta2, n)*prior(theta2), theta2)
    return numer/(5*denom)

y = np.arange(4, 50+1)

sns.set()
plt.plot(y, Q(y, prior=unidist), label=r'$\mathrm{Uni}(0,1)$')
plt.plot(y, Q(y, prior=beta28), label=r'$\mathrm{Beta}(2, 8)$')
plt.plot(y, Q(y, prior=evenmix), label=r'$\mathrm{Beta}(2, 8) + \mathrm{Beta}(8, 2)$')
plt.xlabel(r'$y$')
plt.ylabel(r'$Q(y)$')
plt.legend()
plt.savefig("quotient.pdf", type="pdf")
#plt.show()
