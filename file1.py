
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
from scipy.ndimage import gaussian_filter1d

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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

# -----------------------------
# Parameters
# -----------------------------
pressure_atm = 1.0  # pressure in atm

# -----------------------------
# Load HITRAN data
# -----------------------------
HITRan_data = pd.read_csv('68ffa2cd.txt', usecols=[0,1,2], header=0)
HITRan_data.columns = ['Wavenumber', 'Intensity', 'gamma_air']

# Convert to numeric
HITRan_data = HITRan_data.apply(pd.to_numeric, errors='coerce')

# Filter data
filtered_data = HITRan_data[(HITRan_data['Wavenumber'] >= 550) & 
                            (HITRan_data['Wavenumber'] <= 770)]

# -----------------------------
# Extract data
# -----------------------------
nu0 = filtered_data['Wavenumber'].values
S = filtered_data['Intensity'].values
gamma_air = filtered_data['gamma_air'].values

# Scale gamma by pressure
gamma = gamma_air * pressure_atm

# -----------------------------
# Fine wavenumber grid
# -----------------------------
nu = np.linspace(nu0.min() - 5, nu0.max() + 5, 1000)

# -----------------------------
# Compute Lorentzian cross-section
# -----------------------------
sigma = np.zeros_like(nu)
for i in range(len(nu0)):
    L = (1/np.pi) * gamma[i] / ((nu - nu0[i])**2 + gamma[i]**2)
    sigma += S[i] * L

# -----------------------------
# Method of Least Squares Triangles in Log-Log Space
# -----------------------------
mask = sigma > 0
nu_pos = nu[mask]
sigma_pos = sigma[mask]

sigma_smooth = gaussian_filter1d(sigma_pos, sigma=5.0)

peak_idx = np.argmax(sigma_smooth)
nu_peak = nu_pos[peak_idx]
sigma_peak = sigma_pos[peak_idx]

log_sigma = np.log10(sigma_pos)
log_peak = log_sigma[peak_idx]

# Use full range (no restriction near the peak)
nu_near = nu_pos
log_near = log_sigma

# Split into left/right sides
peak_idx_near = np.argmin(np.abs(nu_near - nu_peak))
nu_peak_near = nu_near[peak_idx_near]
log_peak_near = log_near[peak_idx_near]

nu_left = nu_near[nu_near < nu_peak_near]
nu_right = nu_near[nu_near >= nu_peak_near]
log_left = log_near[nu_near < nu_peak_near]
log_right = log_near[nu_near >= nu_peak_near]

# Fit slopes (through the vertex)
Xl = nu_left - nu_peak_near
Yl = log_left - log_peak_near
a_left = np.sum(Xl * Yl) / np.sum(Xl**2) if len(Xl) > 1 else 0.0

Xr = nu_right - nu_peak_near
Yr = log_right - log_peak_near
a_right = np.sum(Xr * Yr) / np.sum(Xr**2) if len(Xr) > 1 else 0.0

fit_log_left = a_left * (nu_left - nu_peak_near) + log_peak_near
fit_log_right = a_right * (nu_right - nu_peak_near) + log_peak_near

fit_left = 10**fit_log_left
fit_right = 10**fit_log_right

# -----------------------------
# Nonlinear least squares Lorentzian fit (no restriction)
# -----------------------------
def lorentz(nu, A, nu0, Gamma):
    return (A/np.pi) * (Gamma / ((nu - nu0)**2 + Gamma**2))

# Fit across *entire* dataset where sigma > 0
nu_fit = nu_pos
sigma_fit = sigma_pos

# Initial guesses
A0 = np.trapz(sigma_fit, nu_fit) * np.pi
nu0_0 = nu_fit[np.argmax(sigma_fit)]
Gamma0 = 1.0
p0 = [A0, nu0_0, Gamma0]

# Perform fit
popt, pcov = curve_fit(lorentz, nu_fit, sigma_fit, p0=p0, maxfev=10000)
A_fit, nu0_fit, Gamma_fit = popt

# Compute fitted Lorentzian
sigma_lor_fit = lorentz(nu, *popt)

# -----------------------------
# Plotting
# -----------------------------
plt.figure(figsize=(10, 6))
plt.xlim(600, 740)
plt.ylim(1e-22, 1e-17)
plt.yscale('log')

plt.plot(nu, sigma, color='blue', lw=1.5, label='Lorentzian cross-section',zorder=1)
plt.scatter(nu0, S, color='red', s=5, alpha=0.8, label='HITRAN lines', zorder=2)
plt.plot(nu_left, fit_left, color='yellow',linestyle= '--', lw=3, zorder=3)
plt.plot(nu_right, fit_right, color='yellow', linestyle= '--', lw=3, zorder=3, label='Least Squares fit')
plt.plot(nu, sigma_lor_fit, color='green', linestyle= '--',lw=2, label='Full LS Lorentz fit', zorder=4)

plt.xlabel('Wavenumber (cm⁻¹)')
plt.ylabel('Absorption cross-section (cm²/molecule)', color='blue')
plt.legend()
plt.tick_params(axis='y', labelcolor='blue')
plt.show()


#####################################
#####################################



####################################################
###############################################
###############################################

def planck_law_lambda_um(wavelength_um, T):
    wavelength_m = wavelength_um * 1e-6
    exponent = (h * c) / (wavelength_m * k * T)
    B = (2 * h * c**2) / (wavelength_m**5 * (np.exp(exponent) - 1))
    return B * 1e-6  # Convert from per m to per µm


def optical_depth(NumberDensity, CrossSection, PathLength,CO2ppm):
    """Calculate optical depth."""
    return NumberDensity * CrossSection * PathLength * CO2ppm/350


# Wavelength range (µm)
lam_um = np.linspace(0.1, 50, 100000)

lam_sigma_um = 1e4 / nu  # convert to µm

sigma_interp = np.interp(lam_um, lam_sigma_um[::-1], sigma[::-1])

z0= 8000
CO2ppm = 400  # current CO2 concentration in ppm
N0 = 1
OD_lamda = optical_depth(N0, sigma_interp, z0, CO2ppm)

def single_slab_radiative_transfer(OpticalDepth, Temp_surface, T_Trop, wavelength):
    """Calculate outgoing spectral irradiance using single-slab radiative transfer."""
    x = planck_law_lamda(wavelength, Temp_surface) * np.exp(-OpticalDepth)
    y = planck_law_lamda(wavelength, T_Trop) * (1 - np.exp(-OpticalDepth))
    return x + y


T_surface = 288
Gamma_LR = 0.00649       # K/m
eta = 0.75
T_trop = T_surface - Gamma_LR * z0 * np.log(1 - eta)

I_out = single_slab_radiative_transfer(OD_lamda, T_surface, T_trop, lam_um)


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



# Temperatures
T_sun = 5770
T_earth = 254.9

# Outgoing (Earth emission)
E_lambda_out = planck_law_lambda_um(lam_um, T_earth)

plt.figure(figsize=(10,6))
plt.plot(lam_um, planck_law_lambda_um(lam_um, T_surface), label='No CO2 absorption')
plt.plot(lam_um, I_out, label='With CO2 absorption')
plt.xlim(0, 20)  # zoom on CO2 band for better visibility
plt.xlabel('Wavelength (µm)')
plt.ylabel('Spectral Irradiance (W/m²/µm)')
plt.title('Outgoing Radiation Spectrum with CO2 Notch')
plt.legend()
plt.show()