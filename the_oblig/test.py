import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def f(y, theta):
    return theta/(2*np.sqrt(y))*np.exp(-theta*np.sqrt(y))

theta = 1

y = np.linspace(1, 1000, 100000)
