import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
import arviz as az
import theano.tensor as tt

x = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
x = np.array(x)

y = [2, 1, 2, 2, 3, 4, 1, 2, 5, 3]
y = np.array(y)

m = len(y)

N_samples = 20000

with pm.Model() as model:
    a = pm.Uniform('a', lower=-8, upper=8)
    b = pm.Uniform('b', lower=-8, upper=8)
    
    p = pm.Deterministic('p', tt.exp(a + b*x)/(1 + tt.exp(a + b*x)))
    
    observed = pm.Binomial('observed', m, p, observed=y)
    
    step = pm.Metropolis()
    
    trace = pm.sample(N_samples, step=step)