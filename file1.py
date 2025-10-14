

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad


def gaussian(x, mu=0, sigma=1):
    """Calculate the Gaussian function."""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def plot_gaussian(mu=0, sigma=1, xlim=(-5, 5), num_points=1000):
    """Plot the Gaussian function."""
    x = np.linspace(xlim[0], xlim[1], num_points)
    y = gaussian(x, mu, sigma)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=f'Gaussian (μ={mu}, σ={sigma})')
    plt.title('Gaussian Function')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.axhline(0, color='black', lw=0.5, ls='--')
    plt.axvline(0, color='black', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()

plot_gaussian(mu=0, sigma=1)
print("addition")
print("subtraction")