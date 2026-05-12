# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:10:14 2024

@author: Diddams
"""

# %% Imports

import os, copy
import numpy as np
rng = np.random.default_rng()

from scipy import interpolate, signal
from scipy.constants import c, pi, h
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = (10**8)
from matplotlib import pyplot as plt, ticker
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['grid.alpha'] = 0.25
plt.rcParams['savefig.dpi'] = 600
import matplotlib.ticker as mticker
import pynlo 
# from pynlo.medium import RamanResponse
# from pynlo.utility import fft

# %% Pulse

v_min = c/4000e-9
v_max = c/400e-9
v0 = c/1560e-9
e_p = 50e-12 
# e_p = 3.5e-11
t_fwhm = 210e-15
# t_fwhm = 50e-15

T0 = t_fwhm / 1.763
P0_expected = e_p / T0
print("Expected P0 (W):", P0_expected)
phi_NL = 1.3 * P0_expected * 0.01
print("Nonlinear phase shift (rad):", phi_NL)


#JULIA ADDED:
# dv = 300e12
# v_min = v0 - dv
# v_max = v0 + dv

n_points = 2**13 # 20 for sidebands

pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
print("Frq Res: {:.3g} GHz".format(pulse.dv * 1e-9))
v_grid = pulse.v_grid

#%% SiN waveguide
thickness = 600e-9 # 420, 350
width = 1200e-9 # 1300, 1800
import ri_interpolator
sim_freqs = ri_interpolator.sim_freqs
sim_oversample = np.linspace(sim_freqs.min(), sim_freqs.max(), sim_freqs.size*100)
sim_n_eff, sim_gamma, sim_a_eff = ri_interpolator.refractive_index_and_gamma(
    [thickness], [width], sim_freqs, mode='Ex')
print(f'sim_gamma = {np.mean(sim_gamma)}')
# sim_gamma = 10.5
gamma_spline = interpolate.InterpolatedUnivariateSpline(
    sim_freqs,
    sim_gamma,
    ext="extrapolate")
n_eff_spline = interpolate.InterpolatedUnivariateSpline(
sim_freqs,
sim_n_eff,
ext="extrapolate",
k=3)

g3_v = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma_spline(v_grid))

beta_v = pynlo.utility.chi1.n_to_beta(v_grid, n_eff_spline(v_grid))
print(f'beta_v = {np.mean(beta_v)}')
domega = np.mean(np.diff(v_grid))
print("mean Δω:", domega)
omega = 2 * np.pi * pulse.v_grid   # angular frequency [rad/s]
print('mean omega:', np.mean(omega))

omega = 2*np.pi * pulse.v_grid  # angular frequency [rad/s]

# beta_w = pynlo.utility.chi1.n_to_beta(omega, n_eff_spline(omega))

# g3_w = pynlo.utility.chi3.gamma_to_g3(omega, gamma_spline(omega))


# # beta_v = np.zeros(len(v_grid))
# print(np.average(beta_v))
# print("LD =", T0**2 / abs(np.average(beta_v)))

#---- Mode
mode = pynlo.medium.Mode(v_grid, beta_v, alpha = None, g3=g3_v)
beta2 = mode.beta2
print("β₂:", np.mean(beta2))
print("LD =", T0**2 / abs(np.average(beta2)))

length = 0.01

#%%
#---- Run Sim
sim = pynlo.model.NLSE(pulse, mode) # NLSE
#---- Estimate step size
local_error = 1e-6
dz = sim.estimate_step_size(local_error=local_error)
# dz = length / 2000   # 2000 steps over 5 mm → 2.5 µm steps #JULIA REDEFINED THIS

new_pulse, z, a_t, a_v = sim.simulate(length, dz=dz, local_error=local_error, n_records=100, plot="frq")

# %% Plot Results

#from matplotlib import colormaps as cm
"""
For comparison with Dudley, we plot the evolution in the time and wavelength
domains. For accurate representation of the density, plotting over wavelength
requires converting the power from a per Hz basis to a per m basis. This is
accomplished by multiplying the frequency domain power spectral density by the
ratio of the frequency and wavelength differentials. The power spectral density
is then converted to decibel (dB) scale to increase the visible dynamic range.

"""
fig = plt.figure("Simulation Results", clear=True, figsize = (12,8.5))
fig.subplots_adjust(hspace=0.4)
ax0 = plt.subplot2grid((3,2), (0, 0), rowspan=1)
ax1 = plt.subplot2grid((3,2), (0, 1), rowspan=1)
ax2 = plt.subplot2grid((3,2), (1, 0), rowspan=2, sharex=ax0)
ax3 = plt.subplot2grid((3,2), (1, 1), rowspan=2, sharex=ax1)

p_l_dB = 10*np.log10(np.abs(a_v)**2 * sim.dv_dl)
p_l_dB -= p_l_dB.max()
# ax0.plot(1e9*c/pulse.v_grid, p_l_dB[0], color="b")
# ax0.plot(1e9*c/pulse.v_grid, p_l_dB[-1], color="g")
# ax2.pcolormesh(1e9*c/pulse.v_grid, 1e3*z, p_l_dB,
#                 vmin=-40.0, vmax=0, shading="auto")
ax0.plot(1e-12*pulse.v_grid, p_l_dB[0], color="b", label = 'Input')
ax0.plot(1e-12*pulse.v_grid, p_l_dB[-1], color="g", label = 'Output')
im = ax2.pcolormesh(1e-12*pulse.v_grid, 1e3*z, p_l_dB,
                vmin=-52.0, vmax=0, shading="auto", cmap = 'nipy_spectral')
ax0.set_ylim(bottom=-90, top=5)
# ax0.set_xlim(left = 50, right=470)
ax0.set_xlim(left = 150, right=300)
ax0.legend()
ax2.set_xlabel('Frequency (THz)')

p_t_dB = 10*np.log10(np.abs(a_t)**2)
p_t_dB -= p_t_dB.max()
ax1.plot(1e12*pulse.t_grid, np.abs(a_t[0])**2/np.max( np.abs(a_t[0])**2), color="b", label = 'Input')
ax1.plot(1e12*pulse.t_grid, np.abs(a_t[-1])**2/np.max( np.abs(a_t[-1])**2), color="g", label = 'Output')
ax1.plot(1e12*pulse.t_grid, np.abs(a_t[12])**2/np.max( np.abs(a_t[12])**2), color="r", label = 'Bubbles')
im = ax3.pcolormesh(1e12*pulse.t_grid, 1e3*z, p_t_dB,
                vmin=-57.0, vmax=0, shading="auto", cmap = 'nipy_spectral')
ax1.set_ylim(bottom=-0.1, top=1.1)
ax1.set_xlim(left = -0.25, right=0.25) #0.6
# ax1.set_xlim(left=-2.0, right=2.0)
ax1.legend()
ax3.set_xlabel('Time (ps)')
cb_ax = fig.add_axes([.91,.125,.02,.454])
fig.colorbar(im,orientation='vertical',cax=cb_ax)

ax0.set_ylabel('Intensity (dB)')
ax2.set_ylabel('Length (mm)', labelpad = 20)
plt.show()


#%% Adding in Julia's Plots:
t = pulse.t_grid
I_out = np.abs(a_t[-1])**2

# plt.plot(t*1e12, I_out)
# plt.axhline(0, color='k')
# plt.xlabel("Time (ps)")
# plt.ylabel("Intensity")
# plt.title("Output temporal profile")
# plt.show()

margin = 0.1 * np.ptp(pulse.t_grid)
edge_mask = np.abs(pulse.t_grid) > (np.ptp(pulse.t_grid)/2 - margin)

print("Max edge intensity:",
      np.max(I_out[edge_mask]) / np.max(I_out))

# print(f'length of the array = {len(a_t)}')

# Temporal field
a_t = pulse.a_t
t = pulse.t_grid * 1e12  # ps

I_t = np.abs(a_t)**2
phase_t = np.unwrap(np.angle(a_t))

# Mask low-intensity regions (e.g. below -40 dB)
mask_t = I_t > I_t.max() * 1e-4

# Spectral field
dt = pulse.t_grid[1] - pulse.t_grid[0]

# FFT with correct centering
a_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a_t))) * dt
v = pulse.v_grid * 1e-12  # THz

I_w = np.abs(a_w)**2
phase_w = np.unwrap(np.angle(a_w))

# Mask weak spectral components
# mask_w = I_w > I_w.max() * 1e-4

# plt.figure()
# plt.plot(t[mask_t], phase_t[mask_t])
# plt.xlabel("Time (ps)")
# plt.ylabel("Phase (rad)")
# plt.title("Temporal phase")
# plt.grid(True)

# plt.figure()
# plt.plot(v[mask_w], phase_w[mask_w])
# plt.xlabel("Frequency (THz)")
# plt.ylabel("Phase (rad)")
# plt.title("Spectral phase")
# plt.grid(True)
# plt.show()

# # Remove linear phase (fit and subtract)
# p = np.polyfit(v[mask_w], phase_w[mask_w], 1)
# phase_w_rel = phase_w - np.polyval(p, v)

# plt.figure()
# plt.plot(v[mask_w], phase_w_rel[mask_w])
# plt.xlabel("Frequency (THz)")
# plt.ylabel("Residual phase (rad)")
# plt.title("Spectral phase (carrier removed)")
# plt.grid(True)
# plt.show()
