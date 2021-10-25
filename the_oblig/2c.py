import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma

a = 4.4
b = 2.2

y = [0.771, 0.140, 0.135, 0.007, 0.088, 0.008, 0.268, 0.022, 0.131, 0.142, 0.421, 0.125]
y = np.array(y)

n = len(y)

w_n = np.mean(np.sqrt(y))

prior = gamma(a=a, scale=1/b)

posterior = gamma(a=a + n, scale=1/(b + n*w_n))

theta = np.linspace(0, 6, 1000)

# sns.set()
# plt.plot(theta, prior.pdf(theta), label='Prior')
# plt.plot(theta, posterior.pdf(theta), label='Posterior')
# plt.xlabel(r'$\theta$')
# plt.legend()
# plt.savefig('2c.pdf', type='pdf')
# plt.show()
#
# intervals = [0, 1.50, 3.00, 100]
#
columns = ['Prior', 'Posterior']
# rows = ['(0, 1.50)', '(1.50, 3.00)', '(3.00, ∞)']
#
# probs = np.empty((3, 2))
#
# for i in range(3):
#     probs[i, 0] = prior.cdf(intervals[i+1]) - prior.cdf(intervals[i])
#     probs[i, 1] = posterior.cdf(intervals[i+1]) - posterior.cdf(intervals[i])
#
# probs = probs.round(2)
# probs = pd.DataFrame(probs)
# probs.columns = columns
# probs.index = rows
#
# print(probs)

@np.vectorize
def L_A(theta):
    if theta <= 1.50:
        return 0
    else:
        return 1

@np.vectorize
def L_B(theta):
    if 1.50 < theta < 3.00:
        return 0
    else:
        return 2

@np.vectorize
def L_C(theta):
    if theta > 3.00:
        return 0
    else:
        return 3

expected_loss = np.zeros((3, 2))

funcs = [L_A, L_B, L_C]

for i in range(3):
    expected_loss[i, 0] = prior.expect(funcs[i])
    expected_loss[i, 1] = posterior.expect(funcs[i])

expected_loss = expected_loss.round(2)
expected_loss = pd.DataFrame(expected_loss)
expected_loss.columns = columns
expected_loss.index = ['A', 'B', 'C']

print(expected_loss)
