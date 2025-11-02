
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


###############################
###############################

# -----------------------------
# Co-Pilot Effort: Outgoing Spectrum with CO2 Absorption
# -----------------------------


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import h, c, k, N_A

# --- Planck function per µm
def planck_per_um(lam_um, T):
    lam_m = lam_um * 1e-6
    a = 2.0 * h * c**2
    x = (h * c) / (lam_m * k * T)
    B_m = a / (lam_m**5 * (np.exp(x) - 1.0))   # W m^-2 sr^-1 m^-1
    return B_m * 1e-6                           # W m^-2 sr^-1 µm^-1

# --- Outgoing spectrum (single-slab troposphere) simplified
def outgoing_spectrum(nu_hitr_cm, sigma_hitr_cm2pmol,
                      CO2_ppm=400, T_surf=288,
                      Gamma_LR=0.00649, eta=0.75,
                      lam_min=0.1, lam_max=50.0, nlam=30000):
    """
    Compute outgoing spectrum at TOA with and without CO2 absorption.
    nu_hitr_cm: wavenumber grid (cm^-1)
    sigma_hitr_cm2pmol: absorption cross-section (cm^2/molecule)
    Returns: lam_grid_um, F_clear, F_with_CO2
    """

    # 1) Wavelength grid (µm)
    lam_grid_um = np.linspace(lam_min, lam_max, nlam)

    # 2) Interpolate sigma onto wavelength grid
    lam_um_from_wn = 1e4 / nu_hitr_cm  # cm^-1 -> µm
    interp_sigma = interp1d(lam_um_from_wn, sigma_hitr_cm2pmol, 
                            bounds_error=False, fill_value=0.0)
    sigma_grid = interp_sigma(lam_grid_um) * 1e-4  # convert cm² -> m²

    # 3) Compute effective troposphere height z0 and T_trop
    m_air = 28.97e-3 / N_A  # kg per molecule
    g = 9.80665              # m/s²
    z0 = (k * T_surf) / (m_air * g)
    T_trop = T_surf + Gamma_LR * z0 * np.log(1 - eta)

    # 4) CO2 number density at surface
    x_co2 = CO2_ppm * 1e-6
    p_surface = 101325.0
    N0 = (x_co2 * p_surface) / (k * T_surf)  # molecules/m³

    # 5) Optical depth
    OD = N0 * sigma_grid * z0

    # 6) Single-slab radiative transfer (eqn 6)
    B_surf = planck_per_um(lam_grid_um, T_surf)
    B_trop = planck_per_um(lam_grid_um, T_trop)
    I_toa = B_surf * np.exp(-OD) + B_trop * (1 - np.exp(-OD))

    # Convert to spectral flux (W m^-2 µm^-1)
    F_clear = np.pi * B_surf
    F_with_CO2 = np.pi * I_toa

    return lam_grid_um, F_clear, F_with_CO2


lam, F_clear, F_with = outgoing_spectrum(nu, sigma, CO2_ppm=400.0, T_surf=288.0)

plt.figure(figsize=(9,5))
plt.plot(lam, F_clear, label='No CO2 (clear)', color='C0')
plt.plot(lam, F_with, label='With CO2 (400 ppm)', color='C1', lw=0.8)
plt.xlim(12, 20)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Spectral flux at TOA (W m$^{-2}$ µm$^{-1}$)')
plt.title('Outgoing irradiance (single-slab isothermal troposphere)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# -----------------------------
# James Effort: Outgoing Spectrum with CO2 Absorption
# -----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
h = 6.62607015e-34    # J*s
c = 299792458.0       # m/s
k = 1.380649e-23      # J/K
g = 9.80665            # m/s^2
N_A = 6.02214076e23    # mol^-1

# -----------------------------
# Load HITRAN data
# -----------------------------
HITRAN_data = pd.read_csv('68ffa2cd.txt', usecols=[0, 1, 2], header=0)
HITRAN_data.columns = ['Wavenumber', 'Intensity', 'gamma_air']
HITRAN_data = HITRAN_data.apply(pd.to_numeric, errors='coerce')

nu0 = HITRAN_data['Wavenumber'].values  # in cm^-1
S = HITRAN_data['Intensity'].values

# convert wavenumber → wavelength
lambda_m = 1 / (nu0 * 100)  # meters
lambda_um = lambda_m * 1e6  # microns

# dummy cross-section (replace with your σ(ν) from earlier)
sigma = S * 1e-20  # cm^2/molecule (example scaling)
sigma_m2 = sigma * 1e-4  # convert cm^2 to m^2

# -----------------------------
# Optical Depth Function
# -----------------------------
def optical_depth_lambda(CO2_ppm, sigma_m2, lambda_m):
    M_air = 28.97e-3 / N_A  # kg per molecule
    T_surf = 254.9          # K

    # Correct scale height (m)
    Z0 = (k * T_surf) / (M_air * g)

    # Air number density (molecules/m³)
    N_air = 101325 / (k * T_surf)
    N_CO2 = CO2_ppm * 1e-6 * N_air

    # σ(λ) from σ(ν): σ(λ) = σ(ν)/λ²
    sigma_lambda = sigma_m2 / (lambda_m**2)

    # Optical depth (dimensionless)
    OD = N_CO2 * sigma_lambda * Z0
    return OD

# -----------------------------
# Planck Function (per λ)
# -----------------------------
def planck_law_lambda(lambda_m, T):
    x = (h * c) / (lambda_m * k * T)
    return (2 * h * c**2) / (lambda_m**5 * (np.expm1(x)))

# -----------------------------
# Outgoing spectrum
# -----------------------------
def outgoing_spectrum_CO2(lambda_m, sigma_m2, CO2_ppm=400):
    T_surf = 254.9
    T_trop = 217.0

    OD = optical_depth_lambda(CO2_ppm, sigma_m2, lambda_m)
    B_s = planck_law_lambda(lambda_m, T_surf)
    B_t = planck_law_lambda(lambda_m, T_trop)

    I = B_s * np.exp(-OD) + B_t * (1 - np.exp(-OD))
    return I, OD

# -----------------------------
# Compute and plot
# -----------------------------
I_lambda, OD_lambda = outgoing_spectrum_CO2(lambda_m, sigma_m2, CO2_ppm=400)

plt.figure(figsize=(9,5))
plt.plot(lambda_um, I_lambda*1e-6, color='C2', label='With CO₂ absorption')
plt.xlabel('Wavelength (μm)')
plt.ylabel('Radiance (W m$^{-2}$ sr$^{-1}$ μm$^{-1}$)')
plt.legend()
plt.tight_layout()
plt.show()


# --- Convert sigma to m^2 per molecule
sigma_m2_per_mol = np.array(sigma) * 1e-4   # cm^2 -> m^2

# --- Build regular wavelength grid (µm) for Planck and plotting
lam_min_um = 5.0    # focus around 15 µm band
lam_max_um = 25.0
nlam = 20000
lam_grid_um = np.linspace(lam_min_um, lam_max_um, nlam)
lam_grid_m = lam_grid_um * 1e-6

# --- Interpolate sigma onto wavelength grid
# convert your nu (cm^-1) -> wavelength µm
wn_m = np.array(nu) * 100.0          # cm^-1 -> m^-1
lam_m_from_wn = 1.0 / wn_m           # m
lam_um_from_wn = lam_m_from_wn * 1e6 # µm
# sort for safe interpolation
order = np.argsort(lam_um_from_wn)
interp = interp1d(lam_um_from_wn[order], sigma_m2_per_mol[order],
                  bounds_error=False, fill_value=0.0)
sigma_grid_m2 = interp(lam_grid_um)  # m^2 / molecule on lam_grid

# --- Atmosphere and CO2 number density (use ppm)
def N_co2_ppm_to_number_density(ppm, p=101325.0, T=288.0):
    x = ppm * 1e-6
    return x * p / (k * T)   # molecules per m^3

# --- z0 and troposphere top temperature (ensure T_trop < T_surf)
T_surf = 288.0    # K (set to whatever you want)
g = 9.80665
m_air = 28.97e-3 / N_A   # kg per molecule
z0 = (k * T_surf) / (m_air * g)   # effective column height (m)
Gamma_LR = 0.00649
eta = 0.75
# produce T_trop < T_surf
deltaT = Gamma_LR * z0 * (-np.log(1.0 - eta))   # positive
T_trop = T_surf - deltaT

# --- Planck per µm (W m^-2 sr^-1 µm^-1)
def planck_per_um(lam_um, T):
    lam = lam_um * 1e-6
    x = (h * c) / (lam * k * T)
    B_m = (2.0 * h * c**2) / (lam**5 * (np.exp(x) - 1.0))
    return B_m * 1e-6   # W m^-2 sr^-1 µm^-1

# --- Optical depth OD = N0 * sigma(λ) * z0
CO2_ppm = 400.0
N0_co2 = N_co2_ppm_to_number_density(CO2_ppm, p=101325.0, T=T_surf)  # m^-3
OD = N0_co2 * sigma_grid_m2 * z0      # dimensionless

# --- Check OD signs and ranges
print("z0 (m) =", z0, "T_surf (K) =", T_surf, "T_trop (K) =", T_trop)
print("OD min, median, max:", OD.min(), np.median(OD), OD.max())

# --- Eq (6) directional intensity (per µm)
B_s = planck_per_um(lam_grid_um, T_surf)
B_t = planck_per_um(lam_grid_um, T_trop)
I_toa = B_s * np.exp(-OD) + B_t * (1.0 - np.exp(-OD))

# If you want spectral flux at TOA (W m^-2 µm^-1) multiply by π
F_toa = np.pi * I_toa
F_clear = np.pi * B_s   # no CO2

# --- Plotting (spectral flux)
plt.figure(figsize=(9,5))
plt.plot(lam_grid_um, F_clear, label='No CO2 (clear)', color='C0')
plt.plot(lam_grid_um, F_toa, label=f'With CO2 ({int(CO2_ppm)} ppm)', color='C1')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Spectral flux at TOA (W m$^{-2}$ µm$^{-1}$)')
plt.title('Outgoing irradiance (single-slab isothermal troposphere, eqn (6))')
plt.legend()
plt.grid(alpha=0.3)
plt.show()