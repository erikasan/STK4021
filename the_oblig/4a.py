import numpy as np
from scipy.optimize import minimize
from sympy import symbols, diff

x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
x = np.array(x)

y = [2, 1, 2, 2, 3, 4, 1, 2, 5, 3]
y = np.array(y)

def logistic(x, a, b):
    exp = np.exp(a + b*x)
    return exp/(1 + exp)

def negloglikelihood(a_and_b, *args):
    a, b = a_and_b
    x, y = args
    m = len(y)
    p = logistic(x, a, b)
    return -np.sum(y*np.log(p) + (m - y)*np.log(1 - p))

res = minimize(fun=negloglikelihood, x0=np.array([0,0]), args=(x,y), tol=1e-10)

a_hat, b_hat = res.x

#print(np.array([a_hat, b_hat]).round(3))

a, b = symbols('a b', real=True)
