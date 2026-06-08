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
# phi_NL = 1.3 * P0_expected * 0.01
# print("Nonlinear phase shift (rad):", phi_NL)


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

dt = pulse.dt
r_weights = [0.05, 13.5e-15, 45.0e-15]  # Approximate SiN Raman response
rv_grid, r3 = pynlo.utility.chi3.raman(n=n_points, dt=dt, r_weights=r_weights, b_weights=None, analytic=True) 

print("Raman grid generated. Frequency points:", len(r3))
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
mode = pynlo.medium.Mode(v_grid, beta_v, alpha = None, g3=g3_v, rv_grid=rv_grid, r3=r3)
beta2 = mode.beta2
print("Mean β₂:", np.mean(beta2))
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

def nice_plot(a_v):
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


nice_plot(a_v)

#%% Calculating the Nonlinear phase shift

def nonlin_phas_shift(a_t):
    '''
    Method to calculate the nonlinear phase shift that occurs in the waveguide.
    Params:
        a_t: the electromagnetic spectrum as a function of time.
    Returns:
        None: it plots the nonlinear phase shift and prints a value.
    '''

    # Calculate the raw phase difference
    phase_input = np.unwrap(np.angle(a_t[0]))
    phase_output = np.unwrap(np.angle(a_t[-1]))
    phase_shift_total = phase_output - phase_input

    # Create a mask to ONLY look where the pulse has real power
    # This ignores the chaotic numerical noise at the empty edges
    intensity_input = np.abs(a_t[0])**2
    mask = intensity_input > (np.max(intensity_input) * 1e-3) # Top 30 dB of the pulse

    # Fit and remove the linear frequency shift ONLY within the pulse window
    t_ps = pulse.t_grid * 1e12
    p = np.polyfit(t_ps[mask], phase_shift_total[mask], 1)
    phase_pure_nonlinear = phase_shift_total - np.polyval(p, t_ps)

    # Plot the results focusing only on the physical pulse region
    fig, ax1 = plt.subplots(figsize=(9, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Normalized Intensity', color=color)
    ax1.plot(t_ps, intensity_input / np.max(intensity_input), color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('True Nonlinear Phase Shift (rad)', color=color)
    # Only plot the phase where the pulse is active so it stays clean
    ax2.plot(t_ps[mask], phase_pure_nonlinear[mask], color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.set_xlim(-0.4, 0.4)
    plt.title("Pulse Profile vs. True Nonlinear Phase Shift (Noise Masked)")
    fig.tight_layout()
    plt.show()

    # Print the actual peak value
    peak_idx = np.argmax(intensity_input)
    print(f"True nonlinear phase shift at the peak: {phase_pure_nonlinear[peak_idx]:.2f} rad")


nonlin_phas_shift(a_t)

#%% Creating a second pulse and interfering the two pulses

def pulse_interference(a_v):
    '''
    Create a new un-propagated LO pulse and interfere it with the pulse that has propagated through the waveguide 
    to simulate homodyne detection with a single detector.
    Params:
        a_v: the electromagnetic spectrum defined in frequency.
    Returns:
        None: Plots the interference.
    '''

    # and a_v_aux is the newly created auxiliary pulse object
    # I'll create it here

    e_p_aux = e_p*100
    pulse_aux = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p_aux, t_fwhm)
    a_v_aux = pulse_aux.a_v

    # Define your tuning parameters
    tau = 150e-15      # Time delay in seconds (e.g., 200 fs)
    theta = 1  # Relative global phase shift in radians

    # Apply the delay and phase shift to the auxiliary pulse
    # The time delay tau creates a phase shift of 2*pi*v*tau across the spectrum
    phase_ramp = np.exp(1j * 2 * np.pi * pulse.v_grid * tau)
    global_phase = np.exp(1j * theta)

    a_v_aux_shifted = a_v_aux * global_phase * phase_ramp

    # Interfere them (simply add the complex fields)
    a_v_interfered = a_v[-1] + a_v_aux_shifted

    # Calculate spectral intensities for plotting
    I_main_out = np.abs(a_v[-1])**2
    I_aux = np.abs(a_v_aux_shifted)**2
    I_interfered = np.abs(a_v_interfered)**2

    plt.figure(figsize=(10, 6))
    freq_thz = pulse.v_grid * 1e-12

    # Convert to dB scale for scannability
    def to_db(x): 
        return 10 * np.log10(x / np.max(I_interfered))

    plt.plot(freq_thz, to_db(I_interfered), color='black', linewidth=2, label='Interfered Spectrum')
    plt.plot(freq_thz, to_db(I_main_out), color='tab:green', linestyle='--', alpha=0.7, label='SQZ')
    plt.plot(freq_thz, to_db(I_aux), color='tab:orange', linestyle=':', alpha=0.7, label=rf'AUX, $\theta = ${theta}$\pi$')

    plt.xlabel('Frequency (THz)')
    plt.ylabel('Relative Intensity (dB)')
    plt.title('Spectral Interference (Homodyne Mixing Profile)')
    plt.xlim(150, 250) # Focus on your pulse bandwidth
    plt.ylim(-40, 5)
    plt.grid(True)
    plt.legend()
    plt.show()

pulse_interference(a_v)

#%% Adding in Julia's Plots:
# t = pulse.t_grid
# I_out = np.abs(a_t[-1])**2

# plt.plot(t*1e12, I_out)
# plt.axhline(0, color='k')
# plt.xlabel("Time (ps)")
# plt.ylabel("Intensity")
# plt.title("Output temporal profile")
# plt.show()

# margin = 0.1 * np.ptp(pulse.t_grid)
# edge_mask = np.abs(pulse.t_grid) > (np.ptp(pulse.t_grid)/2 - margin)

# print("Max edge intensity:",
#       np.max(I_out[edge_mask]) / np.max(I_out))

# # print(f'length of the array = {len(a_t)}')

# # Temporal field
# a_t = pulse.a_t
# t = pulse.t_grid * 1e12  # ps

# I_t = np.abs(a_t)**2
# phase_t = np.unwrap(np.angle(a_t))

# # Mask low-intensity regions (e.g. below -40 dB)
# mask_t = I_t > I_t.max() * 1e-4

# # Spectral field
# dt = pulse.t_grid[1] - pulse.t_grid[0]

# # FFT with correct centering
# a_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a_t))) * dt
# v = pulse.v_grid * 1e-12  # THz

# I_w = np.abs(a_w)**2
# phase_w = np.unwrap(np.angle(a_w))

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
