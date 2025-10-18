
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import pandas as pd


h = 6.626e-34  # Planck's constant (J·s)
c = 3e8         # Speed of light (m/s)
k = 1.38e-23    # Boltzmann's constant (J/K)

T = 200 # Temperature in Kelvin


def planck_law_freq(frequency, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * frequency) / (k * temperature)
    return (2 * h * frequency**3) / (c**2 * (np.exp(exponent) - 1))


def planck_law_lamda(wavelength, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * c) / (wavelength * k * temperature)
    return (2 * h * c**2) / (wavelength**5 * (np.exp(exponent) - 1))

graphfrequencies = np.linspace(1e13, 3e15, 10000)

graphtemperatures = np.full(10000, T)

graphradiance = planck_law_freq(graphfrequencies, graphtemperatures)

#Plotting Planck spectrum for given temperature

plt.figure(figsize=(10, 6))
plt.plot(graphfrequencies, graphradiance, label=f'T = {T} K')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·Hz⁻¹)')
plt.title('Black Body Radiation Spectrum')
plt.legend()
plt.show()

####################################################
####################################################




def planck_law_freq(frequency, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * frequency) / (k * temperature)
    return (2 * h * frequency**3) / (c**2 * (np.exp(exponent) - 1))


def planck_law_lamda(wavelength, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * c) / (wavelength * k * temperature)
    return (2 * h * c**2) / (wavelength**5 * (np.exp(exponent) - 1))

T_in = np.full(10000, 5770)
T_out = np.full(10000, 255)

v_sun = np.linspace(1e14, 1e15, 10000)
v_earth = np.linspace(1e13, 1e14, 10000)
lamda_sun = np.linspace(0.1e-6, 3e-6, 10000)
lamda_earth = np.linspace(3e-6, 30e-6, 10000)

B_freq_in = planck_law_freq(v, T_in)
B_freq_out = planck_law_freq(v, T_out)

B_lamda_in = planck_law_lamda(lamda, T_in)
B_lamda_out = planck_law_lamda(lamda, T_out)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].plot(v, B_freq_in, label=f'Incoming T={5770} K')
axes[0,0].plot(v, B_freq_out, label=f'Outgoing T={255} K')
axes[0,0].set_xlabel('Frequency (Hz)')
axes[0,0].set_ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·Hz⁻¹)')
axes[0,0].set_title('Linear scale - Frequency Domain')
axes[0,0].legend()

axes[0,1].plot(lamda*1e9, B_lamda_in, label=f'Incoming T={5770} K')
axes[0,1].plot(lamda*1e9, B_lamda_out, label=f'Outgoing T={255} K')
axes[0,1].set_xlabel('Wavelength (nm)')
axes[0,1].set_ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·m⁻¹)')
axes[0,1].set_title('Linear scale - Wavelength Domain')
axes[0,1].legend()

axes[1,0].loglog(v, B_freq_in, label=f'Incoming T={5770} K')
axes[1,0].loglog(v, B_freq_out, label=f'Outgoing T={255} K')
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].set_ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·Hz⁻¹)')
axes[1,0].set_title('Log-Log scale - Frequency Domain')
axes[1,0].legend()

axes[1,1].loglog(lamda*1e9, B_lamda_in, label=f'Incoming T={5770} K')
axes[1,1].loglog(lamda*1e9, B_lamda_out, label=f'Outgoing T={255} K')
axes[1,1].set_xlabel('Wavelength (nm)')
axes[1,1].set_ylabel('Spectral Radiance (W·sr⁻¹·m⁻²·m⁻¹)')
axes[1,1].set_title('Log-Log scale - Wavelength Domain')
axes[1,1].legend()

plt.tight_layout()
plt.show()



HITRan_data = pd.read_csv('68f37e77.txt',usecols=[0,1,2], header=0)
HITRan.columns= ['Wavenumber', 'Intensity', 'gamma_air']








