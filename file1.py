
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

print(A_out)
print(A_in)

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

plt.figure(figsize=(10, 6))
plt.plot(lamda_um, E_lambda_out, label='Outgoing (254.9 K)')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Spectral Irradiance (W m⁻² µm⁻¹)')
plt.title('Outgoing Black Body Radiation Spectrum at 254.9 K')
plt.show()

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

##############################################################
##############################################################

# Outgoing spectrum (single-slab isothermal troposphere, eqn (6))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.constants import h, c, k, N_A

# -----------------------------
# User inputs / file
# -----------------------------
hitran_file = '68ffa2cd.txt'   # replace if needed
wn_min, wn_max = 400.0, 899.0  # cm^-1 band selection
pressure_atm = 1.0             # atm
CO2_ppm = 400                # ppm to model
T_surf = 254.9                 # surface temperature K (choose 288 K typical)
Gamma_LR = 0.00649             # K/m
eta = 0.75

# -----------------------------
# Load HITRAN-like file (wavenumber [cm^-1], intensity, gamma_air)
# -----------------------------
df = pd.read_csv(hitran_file, usecols=[0,1,2], header=0)
df.columns = ['Wavenumber', 'Intensity', 'gamma_air']
df = df.apply(pd.to_numeric, errors='coerce').dropna()
df = df[(df['Wavenumber'] >= wn_min) & (df['Wavenumber'] <= wn_max)]

nu0 = df['Wavenumber'].values        # cm^-1
S = df['Intensity'].values           # line strength (units assumed consistent)
gamma_air = df['gamma_air'].values
gamma = gamma_air * pressure_atm

# -----------------------------
# Build a high-resolution wavenumber grid and Lorentzian sigma(ν)
# (sigma here will be in same units as S×Lorentz; treat as cm^2/molecule and convert)
# -----------------------------
nu = np.linspace(nu0.min() - 5, nu0.max() + 5, 5000)   # cm^-1
sigma_cm2_per_mol = np.zeros_like(nu)
for i in range(len(nu0)):
    L = (1.0/np.pi) * gamma[i] / ((nu - nu0[i])**2 + gamma[i]**2)
    sigma_cm2_per_mol += S[i] * L

# convert sigma to m^2 per molecule (cm^2 -> m^2)
sigma_m2_per_mol = sigma_cm2_per_mol * 1e-4

# -----------------------------
# Interpolate sigma(λ) onto regular wavelength grid (µm)
# -----------------------------
wn_m = nu * 100.0                    # cm^-1 -> m^-1
lam_m_from_wn = 1.0 / wn_m           # m
lam_um_from_wn = lam_m_from_wn * 1e6 # µm

lam_min_um, lam_max_um = 5.0, 25.0
nlam = 20000
lam_grid_um = np.linspace(lam_min_um, lam_max_um, nlam)
# safe interpolation (sort source)
order = np.argsort(lam_um_from_wn)
interp_sigma = interp1d(lam_um_from_wn[order], sigma_m2_per_mol[order],
                        bounds_error=False, fill_value=0.0)
sigma_grid_m2 = interp_sigma(lam_grid_um)   # m^2 / molecule

# -----------------------------
# Atmospheric column and optical depth
# -----------------------------
# mean molecular mass per molecule for air (kg per molecule)
m_air = 28.97e-3 / N_A
g = 9.80665

# effective column height z0 = k * Ts / (m_air * g)
z0 = (k * T_surf) / (m_air * g)   # m

# troposphere top temperature (ensure T_trop < T_surf)
deltaT = Gamma_LR * z0 * (-np.log(1.0 - eta))
T_trop = T_surf - deltaT

# number density of CO2 at surface (molecules m^-3)
N0_co2 = (CO2_ppm * 1e-6) * 101325.0 / (k * T_surf)

# optical depth OD(λ) = N0 * sigma(λ) * z0
OD = N0_co2 * sigma_grid_m2 * z0

# -----------------------------
# Planck per µm (W m^-2 sr^-1 µm^-1)
# -----------------------------
def planck_per_um(lam_um, T):
    lam = lam_um * 1e-6
    x = (h * c) / (lam * k * T)
    B_m = (2.0 * h * c**2) / (lam**5 * (np.exp(x) - 1.0))
    return B_m * 1e-6

B_surf = planck_per_um(lam_grid_um, T_surf)
B_trop = planck_per_um(lam_grid_um, T_trop)

# Eqn (6) directional intensity at TOA (per µm)
I_toa = B_surf * np.exp(-OD) + B_trop * (1.0 - np.exp(-OD))

# spectral flux at TOA (multiply by π) in W m^-2 µm^-1
F_toa = np.pi * I_toa
F_clear = np.pi * B_surf

# -----------------------------
# Diagnostics
# -----------------------------
print("Diagnostics:")
print(f"  z0 = {z0:.1f} m, T_surf = {T_surf:.1f} K, T_trop = {T_trop:.2f} K")
print(f"  OD: min = {OD.min():.3e}, median = {np.median(OD):.3e}, max = {OD.max():.3e}")
print(f"  sigma_grid_m2: min = {sigma_grid_m2.min():.3e}, max = {sigma_grid_m2.max():.3e}")

F_total_clear = np.trapz(F_clear, lam_grid_um)
F_total_with  = np.trapz(F_toa, lam_grid_um)
print(f"  Integrated flux (no CO2)   = {F_total_clear:.3f} W/m^2")
print(f"  Integrated flux (with CO2) = {F_total_with:.3f} W/m^2")
print(f"  Delta F = {F_total_clear - F_total_with:.3f} W/m^2")

# -----------------------------
# Plots
# -----------------------------
plt.figure(figsize=(9,5))
plt.plot(lam_grid_um, F_clear, label='No CO2 (clear)', color='C0', lw=1)
plt.plot(lam_grid_um, F_toa, label=f'With CO2 ({int(CO2_ppm)} ppm)', color='C1', lw=1)
plt.xlim(5, 25)
plt.ylim(0, np.max(F_clear[(lam_grid_um>8)&(lam_grid_um<12)])*1.2)
plt.xlabel('Wavelength (µm)')
plt.ylabel('Spectral flux at TOA (W m$^{-2}$ µm$^{-1}$)')
plt.title('Outgoing irradiance (single-slab isothermal troposphere, eqn (6))')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# integrate spectral flux (W m^-2)
F_total_with_CO2 = integrate.simpson(F_toa, lam_grid_um)   # W/m^2
F_total_clear = integrate.simpson(F_clear, lam_grid_um)    # W/m^2

print(f"Integrated flux (no CO2)   = {F_total_clear:.3f} W/m^2")
print(f"Integrated flux (with CO2) = {F_total_with_CO2:.3f} W/m^2")

#solving for earth surface temperature without atmosphere
# Energy balance (no atmosphere):
# (1 - albedo) * ∫ I_nu_sun dν  =  4 * π * ∫ B_nu_earth(T) dν
# where:
#   I_nu_sun : spectral irradiance of the Sun at Earth (W/m²/Hz)
#   B_nu_earth(T) : Planck spectral radiance of Earth at temperature T (W/m²/sr/Hz)
#   albedo : fraction of sunlight reflected by Earth

albedo = 0.296
Leftside = (1 - albedo) * A_in

Rightside = 4*np.pi*

print(Leftside)
print(Rightside)