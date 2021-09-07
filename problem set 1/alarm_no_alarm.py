import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def model(y, theta, n):
    return theta**y*(1 - theta)**(n - y)


n = 50

def Q(y):
    theta1 = np.linspace(0, 0.15, 150)
    theta2 = np.linspace(0.15, 1, 1000 - 150)
    numer = np.trapz(model(y, theta1, n), theta1)
    denom = np.trapz(model(y, theta2, n), theta2)
    return numer/(5*denom)
