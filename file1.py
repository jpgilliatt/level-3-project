
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
from scipy.special
import pandas as pd


h = 6.626e-34  # Planck's constant (J·s)
c = 3e8         # Speed of light (m/s)
k = 1.38e-23    # Boltzmann's constant (J/K)

T = 1000 # Temperature in Kelvin


def planck_law(frequency, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * frequency) / (k * temperature)
    return (2 * h * frequency**3) / (c**2 * (np.exp(exponent) - 1))


def planck_law_wavelength(wavelength, temperature):
    """Calculate the spectral radiance of a black body at a given temperature using wavelength."""
    exponent = (h * c) / (wavelength * k * temperature)
    return (2 * h * c**2) / (wavelength**5 * (np.exp(exponent) - 1))


graphfrequencies = np.linspace(1e12, 1e15, 10000)

graphtemperatures = np.full(10000, T)

graphradiance = planck_law(graphfrequencies, graphtemperatures)

graph1wavelengths = c / graphfrequencies

plt.figure(figsize=(10, 6))
plt.plot(graph1wavelengths *10**9, graphradiance, label=f'T = {T} K')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·Hz⁻¹)')
plt.title('Black Body Radiation Spectrum')
plt.legend()
plt.show()

#Plotting Planck spectrum for given temperature

plt.figure(figsize=(10, 6))
plt.plot(graphfrequencies, graphradiance, label=f'T = {T} K')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·Hz⁻¹)')
plt.title('Black Body Radiation Spectrum')
plt.legend()
plt.show()



HITRan_data = pd.read_csv('68f37e77.txt',usecols=[0,1,2], header=0)
HITRan.columns= ['Wavenumber', 'Intensity', 'gamma_air']

x=np.linspace(550,800,5000) # Wavenumber range in cm^-1





def lorentzian(x, x0, gamma):
    """Calculate the Lorentzian line shape."""
    return (gamma / np.pi) / ((x - x0)**2 + gamma**2)




