
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
###############################################
###############################################

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import integrate
from scipy.optimize import fsolve

# Constants
h = 6.626e-34  # Planck constant (J·s)
c = 3e8        # Speed of light (m/s)
k = 1.38e-23   # Boltzmann constant (J/K)

# Planck's law (per wavelength in µm)
def planck_law_lambda_um(wavelength_um, T):
    wavelength_m = wavelength_um * 1e-6
    exponent = (h * c) / (wavelength_m * k * T)
    B = (2 * h * c**2) / (wavelength_m**5 * (np.exp(exponent) - 1))
    return B * 1e-6  # Convert from per m to per µm

# Scale from Sun surface → Earth orbit
R_sun = 6.96e8  # m
AU = 1.496e11   # m
scale_sun_to_earth = np.pi * (R_sun / AU)**2  # ≈ 6.82×10⁻⁵

# Convert spectral irradiance per λ to per ν
def spectral_lambda_to_freq(E_lambda, wavelength_um):
    wavelength_m = wavelength_um * 1e-6
    return E_lambda * wavelength_m**2 / c

# Wavelength range (µm)
lamda_um = np.linspace(0.1, 500, 100000)
freq = c / (lamda_um * 1e-6)

# Temperatures
T_sun = 5770
T_earth = 254.9

# Incoming (Sun → Earth)
E_lambda_in = planck_law_lambda_um(lamda_um, T_sun) * scale_sun_to_earth
E_freq_in = spectral_lambda_to_freq(E_lambda_in, lamda_um)

# Outgoing (Earth emission)
E_lambda_out = planck_law_lambda_um(lamda_um, T_earth)
E_freq_out = spectral_lambda_to_freq(E_lambda_out, lamda_um)

# Load E490 data
E490Spectrum = pd.read_csv('E490SolarSpectrum.txt', delim_whitespace=True, header=None,
                           names=['Wavelength_micro_m', 'Irradiance_W_m2_micro_m'])
E490Spectrum_lamda = E490Spectrum['Wavelength_micro_m']
E490Spectrum_Irradiance = E490Spectrum['Irradiance_W_m2_micro_m']
E490Spectrum_freq = c / (E490Spectrum_lamda * 1e-6)
E490Spectrum_Irradiance_freq = spectral_lambda_to_freq(E490Spectrum_Irradiance, E490Spectrum_lamda)

# Integrate (area under curves) in wavelength
A_in = integrate.simpson(E_lambda_in, lamda_um)
A_out = integrate.simpson(E_lambda_out, lamda_um)
A_E490 = integrate.simpson(E490Spectrum_Irradiance, E490Spectrum_lamda)

#solving for earth surface temperature without atmosphere
# Energy balance (no atmosphere):
# (1 - albedo) * ∫ I_nu_sun dν  =  4 * π * ∫ B_nu_earth(T) dν
# where:
#   I_nu_sun : spectral irradiance of the Sun at Earth (W/m²/Hz)
#   B_nu_earth(T) : Planck spectral radiance of Earth at temperature T (W/m²/sr/Hz)
#   albedo : fraction of sunlight reflected by Earth

albedo = 0.296
Leftside = (1 - albedo) * A_in

Rightside = 4*np.pi*A_out

print(Leftside)
print(Rightside)


# Plotting
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Linear - Frequency
axes[0, 0].plot(freq, E_freq_in, label='Incoming (5770 K)')
axes[0, 0].plot(freq, E_freq_out, label='Outgoing (254.9 K)')
axes[0, 0].plot(E490Spectrum_freq, E490Spectrum_Irradiance_freq,
                label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[0, 0].set_xlim(1e13, 1.5e15)  # Set x-axis limits for better visibility
axes[0, 0].set_xlabel('Frequency (Hz)')
axes[0, 0].set_ylabel('Spectral Irradiance (W m⁻² Hz⁻¹)')
axes[0, 0].set_title('Linear scale — Frequency domain')
axes[0, 0].legend()

# Linear - Wavelength
axes[0, 1].plot(lamda_um, E_lambda_in, label='Incoming (5770 K)')
axes[0, 1].plot(lamda_um, E_lambda_out, label='Outgoing (254.9 K)')
axes[0, 1].plot(E490Spectrum_lamda, E490Spectrum_Irradiance,
                label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[0, 1].set_xlabel('Wavelength (µm)')
axes[0, 1].set_ylabel('Spectral Irradiance (W m⁻² µm⁻¹)')
axes[0, 1].set_title('Linear scale — Wavelength domain')
axes[0,1].set_xlim(0, 3)  # Zoom in for better visibility
axes[0, 1].text(
    0.05, 0.85,
    f"∫ Incoming: {A_in:.1f} W/m²\n"
    f"∫ Outgoing: {A_out:.1f} W/m²\n"
    f"∫ E490: {A_E490:.1f} W/m²",
    transform=axes[0, 1].transAxes,
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none')
)
axes[0, 1].legend()

# Log–log - Frequency
axes[1, 0].loglog(freq, E_freq_in, label='Incoming (5770 K)')
axes[1, 0].loglog(freq, E_freq_out, label='Outgoing (254.9 K)')
#axes[1, 0].loglog(E490Spectrum_freq, E490Spectrum_Irradiance_freq,
#                  label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[1, 0].set_ylim(1e-35, 1e-6)  # Set x-axis limits for better visibility
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('Spectral Irradiance (W m⁻² Hz⁻¹)')
axes[1, 0].set_title('Log–log scale — Frequency domain')
axes[1, 0].legend()

# Log–log - Wavelength
axes[1, 1].loglog(lamda_um, E_lambda_in, label='Incoming (5770 K)')
axes[1, 1].loglog(lamda_um, E_lambda_out, label='Outgoing (254.9 K)')
#axes[1, 1].loglog(E490Spectrum_lamda, E490Spectrum_Irradiance,
#                  label='E490 Solar Spectrum', color='green', alpha=0.5)
axes[1, 1].set_xlim(0.1, 500)  # Set x-axis limits for better visibility
axes[1, 1].set_ylim(1e-5, 1e4)
axes[1, 1].set_xlabel('Wavelength (µm)')
axes[1, 1].set_ylabel('Spectral Irradiance (W m⁻² µm⁻¹)')
axes[1, 1].set_title('Log–log scale — Wavelength domain')
axes[1, 1].legend()

plt.tight_layout()
plt.show()


##########################
##########################


HITRan_data = pd.read_csv('68ffa2cd.txt',usecols=[0,1,2], header=0)
HITRan_data.columns= ['Wavenumber', 'Intensity', 'gamma_air']
print(HITRan_data.head(6))

HITRan_data['Wavenumber'] = pd.to_numeric(HITRan_data['Wavenumber'], errors='coerce')
HITRan_data['Intensity'] = pd.to_numeric(HITRan_data['Intensity'], errors='coerce')
HITRan_data['gamma_air'] = pd.to_numeric(HITRan_data['gamma_air'], errors='coerce')

plt.figure(figsize=(10, 6))

plt.plot(HITRan_data['Wavenumber'], HITRan_data['Intensity'])
plt.xlim(550,770)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('CO2 Absorption Stick Spectrum from HITRAN')
plt.grid()
plt.show()

filtered_data = HITRan_data[(HITRan_data['Wavenumber'] >= 550) & (HITRan_data['Wavenumber'] <= 770)]

# Plot using stem
plt.figure(figsize=(10, 6))
plt.stem(filtered_data['Wavenumber'], filtered_data['Intensity'], linefmt='C0-', markerfmt=' ', basefmt=' ')
plt.xlim(600, 725)
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Intensity')
plt.title('CO2 Absorption Stick Spectrum from HITRAN')
plt.grid()
plt.show()

###########################
###########################

nu_min=filtered_data['Wavenumber'].min()-1
nu_max=filtered_data['Wavenumber'].max()+1
nu_points=np.linspace(nu_min, nu_max,1000)

def lorentzian(nu, nu_0, gamma,S):
    return (S/np.pi) * (gamma / ((nu - nu_0)**2 + gamma**2))

sigma_total = np.zeros_like(nu_points)

plt.figure(figsize=(10, 6))
plt.plot(nu,sigma_total, label='Total Absorption Cross-Section', color='black')
plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorption Cross-Section (cm²)')
plt.title('CO2 Absorption Cross-Section from HITRAN Data')
plt.show()