import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def f(y, theta):
    return theta/(2*np.sqrt(y))*np.exp(-theta*np.sqrt(y))

theta = 1

y = np.linspace(1, 1000, 100000)

def F(theta, n, a):
    return theta**n*np.exp(-theta*a)

n = 2
a = 0.01
theta = np.linspace(0, 10000, 1000000)

sns.set()
plt.plot(theta, F(theta, n, a))
plt.show()
