import numpy as np

def posterior(N):
    return 1/N*(99/100)**(N - 1)

Nmax = 30000
N = np.arange(203, Nmax)
normalization = np.sum(posterior(N))

mean = np.sum(N*posterior(N))/normalization
second_moment = np.sum(N**2*posterior(N))/normalization
variance = second_moment - mean**2
std = np.sqrt(variance)

print("mean = ", mean)
print("std =", std)
