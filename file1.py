
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


E490Spectrum = pd.read_csv('E490SolarSpectrum.txt',delim_whitespace=True, header=None, names=['Wavelength_micro_m', 'Irradiance_W_m2_micro_m'])
E490Spectrum_lamda = E490Spectrum['Wavelength_micro_m']
E490Spectrum_v = c / (E490Spectrum_lamda*1e-6)
E490Spectrum_Irradiance = E490Spectrum['Irradiance_W_m2_micro_m']

plt.plot(E490Spectrum_lamda, E490Spectrum_Irradiance)
plt.show()



####################################################
####################################################




def planck_law_freq(frequency, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    exponent = (h * frequency) / (k * temperature)
    return (2 * h * frequency**3) / (c**2 * (np.exp(exponent) - 1))


def planck_law_lamda_um(wavelength_um, temperature):
    """Calculate the spectral radiance of a black body at a given temperature."""
    wavelength_m = wavelength_um * 1e-6
    exponent = (h * c) / (wavelength_m * k * temperature)
    return (2 * h * c**2) / (wavelength_m**5 * (np.exp(exponent) - 1)) * 1e-6  # Convert to per micrometer




def radiance_freq(B_freq):
    """Convert spectral iradiance to radiance in frequency domain."""
    return np.pi * (6.96e8/1.5e11)**2 * B_freq

def radiance_lamda(B_lamda):
    """Convert spectral iradiance to radiance in wavelength domain."""
    return B_lamda * 6.79e-5

test = planck_law_lamda_um(0.5, 5770)
print(radiance_lamda(test))

T_in = np.full(100000, 5770)
T_out = np.full(100000, 255)

v = np.linspace(1e14, 1e15, 100000)
lamda_um = np.linspace(0.2, 3, 100000)

B_freq_in = planck_law_freq(v, T_in)
Rad_freq_in = radiance_freq(B_freq_in)

B_freq_out = planck_law_freq(v, T_out)
Rad_freq_out = radiance_freq(B_freq_out)

B_lamda_in = planck_law_lamda_um(lamda_um, T_in)
Rad_lamda_in = radiance_lamda(B_lamda_in)

B_lamda_out = planck_law_lamda_um(lamda_um, T_out)
Rad_lamda_out = radiance_lamda(B_lamda_out)

print(Rad_lamda_out)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].plot(v, Rad_freq_in, label=f'Incoming T={5770} K')
axes[0,0].plot(v, Rad_freq_out, label=f'Outgoing T={255} K')
axes[0,0].plot(E490Spectrum_v, E490Spectrum_Irradiance, label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[0,0].set_xlabel('Frequency (Hz)')
axes[0,0].set_ylabel('Spectral Radiance (W·m⁻²·Hz⁻¹)')
axes[0,0].set_title('Linear scale - Frequency Domain')
axes[0,0].legend()

axes[0,1].plot(lamda_um, Rad_lamda_in, label=f'Incoming T={5770} K')
axes[0,1].plot(lamda_um, Rad_lamda_out, label=f'Outgoing T={255} K')
axes[0,1].plot(E490Spectrum_lamda, E490Spectrum_Irradiance, label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[0,1].set_xlabel('Wavelength (um)')
axes[0,1].set_ylabel('Spectral Radiance (W·m⁻²·um⁻¹)')
axes[0,1].set_title('Linear scale - Wavelength Domain')
axes[0,1].legend()

axes[1,0].loglog(v, B_freq_in, label=f'Incoming T={5770} K')
axes[1,0].loglog(v, B_freq_out, label=f'Outgoing T={255} K')
axes[1,0].loglog(E490Spectrum_v, E490Spectrum_Irradiance, label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[1,0].set_xlabel('Frequency (Hz)')
axes[1,0].set_ylabel('Spectral Radiance (W·m⁻²·Hz⁻¹)')
axes[1,0].set_title('Log-Log scale - Frequency Domain')
axes[1,0].legend()

axes[1,1].loglog(lamda_um, B_lamda_in, label=f'Incoming T={5770} K')
axes[1,1].loglog(lamda_um, B_lamda_out, label=f'Outgoing T={255} K')
axes[1,1].loglog(E490Spectrum_lamda, E490Spectrum_Irradiance, label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[1,1].set_xlabel('Wavelength (um)')
axes[1,1].set_ylabel('Spectral Radiance (W·m⁻²·um⁻¹)')
axes[1,1].set_title('Log-Log scale - Wavelength Domain')
axes[1,1].legend()

plt.tight_layout()
plt.show()



####################################################
####################################################


HITRan_data = pd.read_csv('68f37e77.txt',usecols=[0,1,2], header=0)
HITRan_data.columns= ['Wavenumber', 'Intensity', 'gamma_air']








