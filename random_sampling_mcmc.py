import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt

def metropolis_hastings(
    p, 
    dim=2, 
    iterations=1000,
    burn=100,
    thin=1
):
    x = np.random.normal(size=dim)
    samples = []

    for i in range(iterations):
        x_star = x + np.random.normal(size=dim)
        if np.random.rand() < p(x_star) / p(x):
            x = x_star
        if i >= burn and (i - burn) % thin == 0:
            samples.append(x)
    
    return np.array(samples)

if __name__ == '__main__':
    chi_squared = lambda x: chi2.pdf(x, df=55)
    samples = metropolis_hastings(chi_squared, dim=1)
    plt.hist(samples)
    plt.show()