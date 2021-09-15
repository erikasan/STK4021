import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import rv_continuous

class gaussian_gen(rv_continuous):
    def _pdf(self, x):
        return np.exp(-x**2/2)/np.sqrt(2*np.pi)

gaussian = gaussian_gen(name='gaussian')

from scipy.stats import norm
