import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial

y = np.array([6, 8, 7, 6, 7, 4, 11, 8, 6, 3])

a = 0.1
b = 0.1

@np.vectorize
def beta(theta, alpha, beta):
    return beta**alpha*theta**(alpha - 1)*np.exp(-beta*theta)

@np.vectorize
def p(ytilde):
    n = np.size(y)
    ybar = np.mean(y)
    theta = np.linspace(0, 1000, 10000)
    res = np.trapz(beta(theta, alpha=a+n*ybar+ytilde-1, beta=b+n+1), theta)
    return res/factorial(ytilde)

ytilde = np.arange(0, 100)
normalization = np.sum(p(ytilde))
