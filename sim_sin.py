# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 14:10:14 2024

@author: Pooja Sekhar and Julia White
"""

# Imports

import os, copy
import numpy as np
import pandas as pd
seed_value = 42

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
plt.style.use('default')
import pynlo 
import copy
import concurrent.futures
import time
import fiber_sim

# from pynlo.medium import RamanResponse
# from pynlo.utility import fft


def setup_waveguide_and_pulse():
    '''
    Creates the base pulse and waveguide that will be used.
    Params:
        None
    Returns:
        pulse: the pynlo object that represents the pulse
        mode: the pynlo object that represents the waveguide
        v_grid: the frequency grid over which the pulse and mode are created
    '''

    # Pulse
    v_min, v_max, v0 = c/4000e-9, c/400e-9, c/1562e-9
    e_p, t_fwhm = 40e-12, 210e-15
    fib_loss = e_p/(50e-12)
    coup_loss = 1.3
    coup_loss_perc = 10.0 ** (-coup_loss / 10.0)

    n_points = 2**13 # 20 for sidebands

    # pulse = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p, t_fwhm)
    
    # v_grid = pulse.v_grid
    
    pulse = fiber_sim.nd_run()
    pulse.a_v = pulse.a_v * fib_loss * coup_loss_perc
    print("Frq Res: {:.3g} GHz".format(pulse.dv * 1e-9))
    v_grid = pulse.v_grid

    T0 = t_fwhm / 1.763
    P0_expected = e_p / (2*T0)
    print("Expected P0 (W):", P0_expected)

    T0 = t_fwhm / 1.763
    P0_expected = e_p / T0
    print("Expected P0 (W):", P0_expected)

    # SiN waveguide
    thickness, width = 800e-9, 1200e-9
    # import ri_interpolator
    # sim_freqs = ri_interpolator.sim_freqs
    ## -- incorporating numpy mode files from abijith --
    mode_file = 'from_abijith/jw_modes/JW_SiN_O2Clad_800nmThickness_1400nmWidth_gamma_aeff.npy' 
    data = np.load(mode_file)

    # --- 1. Extract and Convert Data
    # Column 0: Wavelength (assumed microns from modesolver.py)
    # Column 1: n_eff
    # Column 2: gamma (1/W/m)
    # Column 3: A_eff (m^2)
    wvl_um = data[:, 0]
    n_eff_data = data[:, 1]
    gamma_data = data[:, 2] 

    # Convert to SI units for frequency mapping
    wvl_m = wvl_um * 1e-6
    freq_data = c / wvl_m

    # --- 2. Sort by Frequency 
    # Splines require the x-axis (frequency) to be strictly increasing.
    # Since wavelength increases, frequency decreases, so we must flip them.
    sort_idx = np.argsort(freq_data)
    freq_data = freq_data[sort_idx]
    n_eff_data = n_eff_data[sort_idx]
    gamma_data = gamma_data[sort_idx]

    # Splines require the x-axis (frequency) to be strictly increasing.
    # Since wavelength increases, frequency decreases, so we must flip them.
    sort_idx = np.argsort(freq_data)
    freq_data = freq_data[sort_idx]
    n_eff_data = n_eff_data[sort_idx]
    gamma_data = gamma_data[sort_idx]

    # --- 3. Create Splines
    # These will map the solver data onto your simulation's v_grid
    n_eff_spline = interpolate.InterpolatedUnivariateSpline(
        freq_data, n_eff_data, k=3, ext="extrapolate")

    gamma_spline = interpolate.InterpolatedUnivariateSpline(freq_data, gamma_data, k=3, ext="extrapolate")

    # --- 4. Setup pynlo Mode
    # beta_v represents the propagation constant
    beta_v = pynlo.utility.chi1.n_to_beta(v_grid, n_eff_spline(v_grid))

    # g3_v represents the third-order nonlinear coupling
    g3_v = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma_spline(v_grid))

    #---- Mode
    length = 0.003
    alpha_val_per_m = (0.1 / 10.0) * np.log(10) / length
    alpha_v = np.full_like(v_grid, alpha_val_per_m) # including loss

    mode = pynlo.medium.Mode(v_grid, beta_v, alpha = None, g3=g3_v)
    # --- Verify Dispersion
    print('')
    beta2 = mode.beta2
    print(f"Mean Beta2 : {np.mean(beta2):.3e} s^2/m")

    # --- Calculate Soliton Period ---
    # Find the exact beta_2 at the central frequency (v0)
    idx_v0 = np.argmin(np.abs(v_grid - v0))
    beta2_v0 = beta2[idx_v0]

    # Calculate Soliton Period (z_0)
    z_0 = np.pi * (t_fwhm/1.7627)**2 / (2 * np.abs(beta2_v0) )

    print(f"Beta2 at v0 ({v0*1e-12:.2f} THz): {beta2_v0:.3e} s^2/m")
    LD = T0**2 / abs(beta2_v0)
    print(f"Dispersion Length (L_D): {LD} m")
    print(f"Soliton Period (z_0): {z_0:.5f} m")
    print(f'Number of soliton periods: {length/z_0}')
    print(f'typical gamma: {np.mean(gamma_data)}')
    n_square = np.mean(gamma_data)*P0_expected*(T0**2)/abs(beta2_v0)
    l_nonlin = 1/(np.mean(gamma_data)*P0_expected)
    print(f'Nonlinear length: {l_nonlin}')
    print(f'n_square = {n_square} or {LD/l_nonlin}')

    return pulse, mode, v_grid

def propagate_pulse(pulse, mode, length=0.003):
    '''
    Propagates the input pulse through the mode (waveguide).
    Params:
        pulse: the input pulse (a pynlo object)
        mode: the waveguide (a pynlo objecgt)
        length: the length of the waveguide
    Returns:
        new_pulse: the output pulse (a pynlo object)
        z:
        a_t: the EM spectrum of the pulse in the time domain
        a_v: the EM spectrum of the pulse in the frequency domain
        sim: the simulation?
    '''

    sim = pynlo.model.NLSE(pulse, mode) # NLSE
    #---- Estimate step size
    local_error = 1e-6
    dz = sim.estimate_step_size(local_error=local_error)
    # dz = length / 2000   # 2000 steps over 5 mm → 2.5 µm steps #JULIA REDEFINED THIS

    new_pulse, z, a_t, a_v = sim.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)
    # change the plot to "frq" if you want it to plot --> I don't want it to do that plot 100+ times so I set plot=None

    return new_pulse, z, a_t, a_v, sim


# Plot Results

#from matplotlib import colormaps as cm

def nice_plot(a_v, sim, pulse, a_t, z):
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
    # ax1.plot(1e12*pulse.t_grid, np.abs(a_t[12])**2/np.max( np.abs(a_t[12])**2), color="r", label = 'Bubbles')
    im = ax3.pcolormesh(1e12*pulse.t_grid, 1e3*z, p_t_dB,
                    vmin=-57.0, vmax=0, shading="auto", cmap = 'nipy_spectral')
    ax1.set_ylim(bottom=-0.1, top=1.1)
    ax1.set_xlim(left = -0.25, right=0.25) #0.6
    # ax1.set_xlim(left=-2.0, right=2.0)
    ax1.legend()
    ax3.set_xlabel('Time (ps)')
    ax3.set_xlim(-0.4,0.4)
    cb_ax = fig.add_axes([.91,.125,.02,.454])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)

    ax0.set_ylabel('Intensity (arb.)')
    ax2.set_ylabel('Length (mm)', labelpad = 20)
    plt.show()


def plot_osa_spectrum(a_v, sim, pulse):
    """
    Plots the spectrum mimicking an OSA output.
    X-axis: Wavelength (nm)
    Y-axis: Intensity (dB/nm)
    """
    # 1. Convert frequency grid to wavelength grid (in nm)
    wvl_nm = (c / pulse.v_grid) * 1e9
    
    # 2. Extract input and output fields
    # sim.dv_dl is the Jacobian converting Power/Hz to Power/m.
    # We multiply by 1e-9 to convert Power/m to Power/nm.
    p_in_per_nm = np.abs(a_v[0])**2 * sim.dv_dl * 1e-9
    p_out_per_nm = np.abs(a_v[-1])**2 * sim.dv_dl * 1e-9
    
    # 3. Convert to dB scale
    # We add a tiny offset (1e-20) to prevent log10(0) warnings
    p_in_dB = 10 * np.log10(p_in_per_nm + 1e-20)
    p_out_dB = 10 * np.log10(p_out_per_nm + 1e-20)

    # Normalize to the input peak to match your experimental plot
    max_dB = np.max(p_in_dB)
    p_in_dB -= max_dB
    p_out_dB -= max_dB
    
    # Optional: If you want RELATIVE intensity (normalized to 0 dB max), uncomment these:
    # max_dB = np.max(p_out_dB)
    # p_in_dB -= max_dB
    # p_out_dB -= max_dB

    # Saving the data to a dataframe
    data = {
        'wavelength': wvl_nm,
        'in': p_in_dB,
        'out': p_out_dB
    }

    df = pd.DataFrame(data)
    df.to_csv('simulated_spectra.csv', index=False)
    
    # 4. Create the Plot
    plt.figure("OSA Spectrum", figsize=(9, 6))
    
    plt.plot(wvl_nm, p_in_dB, color="forestgreen", label="Input", linewidth=2)
    plt.plot(wvl_nm, p_out_dB, color="indigo", label="Output", linewidth=2)
    
    plt.xlabel("Wavelength (nm)", fontsize=12)
    plt.ylabel("Relative Intensity (dB/nm)", fontsize=12)
    plt.title("PyNLO Spectrum", fontsize=18)
    
    # Adjust these limits based on your specific pulse bandwidth
    # plt.xlim(1000, 2200) 
    plt.xlim(1500, 1625)
    
    # Dynamically scale the y-axis to focus on the top 60 dB of the signal
    # plt.ylim(np.max(p_out_dB) - 60, np.max(p_out_dB) + 5)
    
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def nonlin_phas_shift(a_t, pulse, mode, length=0.003):
    '''
    Method to calculate the nonlinear phase shift that occurs in the waveguide.
    Params:
        a_t: the electromagnetic spectrum as a function of time.
    Returns:
        None: it plots the nonlinear phase shift and prints a value.
    '''

    # 1. Get intensity and extract total output phase
    intensity = np.abs(a_t[-1])**2
    max_int = np.max(intensity)
    phase_total = np.unwrap(np.angle(a_t[-1]))
    
    t_ps = pulse.t_grid * 1e12
    peak_idx = np.argmax(intensity)

    # 2. Define a strict mask for the pulse core (e.g., top 15% of intensity)
    # This completely ignores the noisy wings where unwrap goes haywire
    core_mask = intensity > (max_int * 0.15)
    
    # 3. Fit a 2nd-order polynomial (parabola) to the core phase.
    # This represents the linear chirp (dispersion) accumulated by the pulse.
    poly_coefficients = np.polyfit(t_ps[core_mask], phase_total[core_mask], 1)
    linear_chirp_baseline = np.polyval(poly_coefficients, t_ps)
    
    # 4. Subtract the baseline to isolate the pure nonlinear phase shift
    phase_pure_nonlinear = phase_total - linear_chirp_baseline
    
    # Flip the sign if necessary so that a positive intensity yields a positive phase shift plot
    if phase_pure_nonlinear[peak_idx] < 0:
        phase_pure_nonlinear = -phase_pure_nonlinear

    # 5. Plot using a dynamic mask just for clean visualization (top 30 dB)
    vis_mask = intensity > (max_int * 1e-3)

    fig, ax1 = plt.subplots(figsize=(9, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Normalized Output Intensity', color=color)
    ax1.plot(t_ps[vis_mask], intensity[vis_mask] / max_int, color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Pure Nonlinear Phase Shift (rad)', color=color)
    ax2.plot(t_ps[vis_mask], phase_pure_nonlinear[vis_mask], color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("True Nonlinear Phase Shift (Linear Chirp Polyminial Subtracted)")
    fig.tight_layout()
    plt.show()

    print(f"True isolated nonlinear phase shift at the peak: {phase_pure_nonlinear[peak_idx]:.2f} rad")


# Creating a second pulse and interfering the two pulses

def pulse_interference(a_v, pulse):
    '''
    Create a new un-propagated LO pulse and interfere it with the pulse that has propagated through the waveguide 
    to simulate homodyne detection with a single detector.
    Params:
        a_v: the electromagnetic spectrum defined in frequency.
    Returns:
        None: Plots the interference.
    '''

    # Pulse
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

    # and a_v_aux is the newly created auxiliary pulse object
    # I'll create it here

    e_p_aux = e_p/100
    pulse_aux = pynlo.light.Pulse.Sech(n_points, v_min, v_max, v0, e_p_aux, t_fwhm)
    a_v_aux = inject_noise(pulse_aux.a_v, pulse_aux.v_grid, pulse_aux, local_rng=np.random.default_rng(seed_value - 1))
    # a_v_aux = noisy_aux.a_v

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
    plt.title('Spectral Interference')
    plt.xlim(150, 250) # Focus on your pulse bandwidth
    # plt.ylim(-40, 5)
    plt.grid(True)
    plt.legend()
    plt.show()


# Global placeholders that will live inside each separate worker core process
_worker_pulse = None
_worker_mode = None
_worker_v_grid = None
_worker_a_v_lo = None

def init_worker():
    """
    This runs once on each CPU core when the process pool spawns.
    It initializes the un-picklable pynlo objects locally on that core.
    """
    global _worker_pulse, _worker_mode, _worker_v_grid, _worker_a_v_lo, _worker_a_v_clean_out
    
    # Generate the base pulse and mode directly on this core
    _worker_pulse, _worker_mode, _worker_v_grid = setup_waveguide_and_pulse()
    _worker_a_v_lo = copy.deepcopy(_worker_pulse.a_v)
    clean_pulse = copy.deepcopy(_worker_pulse)
    
    # Generate the clean Local Oscillator profile directly on this core
    print("Calculating clean LO pulse on each core...")
    _, _, _, a_v_clean, _ = propagate_pulse(clean_pulse, _worker_mode)
    _worker_a_v_clean_out = a_v_clean[-1]



def inject_noise(a_v, v_grid, pulse, local_rng):
    '''
    Method to add shot noise to the pulse.
    Params:
        a_v: electromagnetic spectrum of the pulse
        v_grid: the frequency spacing of the pulse
        pulse: the simulated pulse
    Returns:
        The electromagnetic spectrum of a pulse that now has noise added to it.
    '''
    # Calculate vacuum energy for each frequency bin
    E_vac = 0.5 * h * v_grid

    # Find the noise amplitude a_v_noise with Energy = sum(|a_v|^2 * dv)
    a_v_noise = np.sqrt(E_vac/pulse.dv)

    # Generate random noise -- this noise needs to be complex
    random_real = local_rng.normal(0,1,len(v_grid))
    random_imag = local_rng.normal(0,1,len(v_grid))

    # # Commented ploting verifies the general Gaussian shape
    # # Code is stolen from https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
    # plt.figure()
    # count, bins, ignored = plt.hist(random_real, 30, density=True)
    # plt.plot(bins, 1/(1 * np.sqrt(2 * np.pi)) *
    #             np.exp( - (bins - 0)**2 / (2 * 1**2) ),
    #         linewidth=2, color='r')
    # plt.show()

    complex_noise = (random_real + (1j * random_imag)) / np.sqrt(2)  #sqrt(2) is there for normalization to split the variance
                                                                    # equally across the real and imaginary axes

    # scale and add to the original pulse field

    return a_v + (a_v_noise * complex_noise)

def verify_vacuum_energy(pulse, v_grid, N=1000):

    energies = np.zeros_like(v_grid)

    for i in range(N):

        rng = np.random.default_rng(i)

        noise = inject_noise(
            np.zeros_like(v_grid,dtype=complex),
            v_grid,
            pulse,
            rng
        )

        energies += np.abs(noise)**2 * pulse.dv

    energies /= N

    expected = 0.5*h*v_grid

    plt.figure(figsize=(8,4))
    plt.plot(v_grid*1e-12,
             energies/expected)

    plt.axhline(1,color='k',ls='--')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Measured / Expected")
    plt.title("Vacuum Energy Verification")
    plt.grid()
    plt.show()


def run_single_iteration(iteration_index):
    if iteration_index == 2:
        s = time.time()

    global _worker_pulse, _worker_mode, _worker_v_grid, _worker_a_v_lo, _worker_a_v_clean_out

    local_rng = np.random.default_rng(seed_value + iteration_index)

    # --- A. Measure pure vacuum noise (Shot Noise Limit Reference) ---
    # Inject noise into a zero-amplitude field
    pure_vacuum_v = inject_noise(np.zeros_like(_worker_v_grid, dtype=complex), _worker_v_grid, _worker_pulse, local_rng)

    # --- B. Measure the propagated noisy signal ---
    # Inject noise into the actual pulse
    noisy_input_v = inject_noise(_worker_pulse.a_v, _worker_v_grid, _worker_pulse, local_rng)

    noisy_pulse = copy.deepcopy(_worker_pulse)
    noisy_pulse.a_v = noisy_input_v
    
    # Propagate the noisy pulse
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(noisy_pulse, _worker_mode, length=.003)
    delta_a_v = a_v_out[-1] - _worker_a_v_clean_out # not sure if this step is necessary but it is technically isolating the noise fluctuations
    I_ref = np.sum(np.abs(a_v_out[-1])**2) * _worker_pulse.dv
    
    if iteration_index == 2:
        print(f'One iteration run time: {time.time() - s} s')

    # Return the results back to the main process
    return I_ref, a_v_out[-1]

# # Running my methods:
# pulse, mode, v_grid = setup_waveguide_and_pulse()
# new_pulse, z, a_t, a_v, sim, v_grid = propagate_pulse(pulse, mode, 0.01)
# noise_pulse = inject_noise(a_v, v_grid, pulse)
# nice_plot(a_v, sim, new_pulse, a_t)
# nonlin_phas_shift(a_t, new_pulse)
# pulse_interference(a_v, new_pulse)

# Simulations with injected noise:
def sim_with_noise_parallel():
    num_iter = 100 # You will likely need 100-1000+ to get clean variance statistics
    
    # 1. Setup everything once
    print("Setting up mode and base pulse...")
    base_pulse, mode, v_grid = setup_waveguide_and_pulse()
    
    # Run one sequential test iteration on the main thread
    print("Running diagnostic single iteration...")
    # Generate the noise spectrum using the pristine base pulse inputs
    noisy_a_v_in = inject_noise(base_pulse.a_v, v_grid, base_pulse, local_rng=np.random.default_rng(seed_value - 1))
    
    # Create a deep copy of the base pulse so we don't modify the master template
    input_pulse = copy.deepcopy(base_pulse)
    
    # Inject the noise into the spectrum BEFORE propagation
    input_pulse.a_v = noisy_a_v_in 
    
    # Propagate the noisy pulse through the waveguide
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(input_pulse, mode, 0.003)
    
    # Safely plot the results on the main thread
    # nice_plot(a_v_out, sim, new_pulse, a_t, z)
    plot_osa_spectrum(a_v_out, sim, new_pulse)
    # verify_vacuum_energy(base_pulse, v_grid, 1000)
    # nonlin_phas_shift(a_t, new_pulse, mode, 0.003)
    # pulse_interference(a_v_out, new_pulse)
    
    # Lists to store the complex overlap integrals
    overlaps_signal = []
    overlaps_vacuum = []
    
    print(f"Starting parallel simulation with {num_iter} iterations...")
    # Use >4 cores (leaving the rest of my PC free so it doesn't freeze up)
    max_cores = 2

    # Start the multiprocessing pool
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores, initializer=init_worker) as executor:
        # Submit all tasks to the executor
        # A list comprehension builds a list of "Futures" (pending tasks)
        futures = [
            executor.submit(run_single_iteration, i)
            for i in range(num_iter)
        ]
        
        # As tasks finish (in any order), gather the results
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                c_vac, c_sig = future.result()
                overlaps_vacuum.append(c_vac)
                overlaps_signal.append(c_sig)
                print(f"Completed iteration {i+1}/{num_iter}")
            except Exception as exc:
                print(f"An iteration generated an exception: {exc}")
                
    # Convert to numpy arrays
    overlaps_signal = np.array(overlaps_signal)
    overlaps_vacuum = np.array(overlaps_vacuum)

    # 2. Sweep the LO phase to find squeezing and anti-squeezing
    phases = np.linspace(0, 4*np.pi, 300)
    var_signal = []
    eta = 0.78

    field_ratio = np.sqrt(150e-6/14.7e-3)

    for theta in phases:

        samples = []

        for field in overlaps_signal:

            aux = field_ratio * base_pulse.a_v * np.exp(1j*theta)

            total = field + aux * .97

            I = np.sum(
                np.abs(total)**2
            ) * base_pulse.dv

            samples.append(I)

        var_signal.append(np.var(samples))

    var_signal = np.array(var_signal)
    print(f'var signal = {np.min(var_signal)}')

    var_ref = np.var(overlaps_vacuum)
    print(f'var ref = {np.mean(var_ref)}')

    var_measured = eta*var_signal + (1-eta)*var_ref
    print(f'var measured = {np.min(var_measured)}')

    squeezing_dB = 10*np.log10(
        var_measured/var_ref
    )

    # 4. Plot the results
    plt.figure(figsize=(8, 5))
    plt.plot(phases, squeezing_dB, label='Output State Noise above SNL', color='indigo', linewidth=2)
    plt.axhline(0, color='k', linestyle='--', label='Shot Noise Limit (SNL)')
    plt.xlabel('Local Oscillator Phase (rad)')
    plt.ylabel('Squeezing (dB)')
    plt.title('Predicted Squeezing vs. LO Phase')
    # plt.xlim(0, 2*np.pi)
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
    
    print(f"Maximum Squeezing: {np.min(squeezing_dB):.2f} dB")
    print(f"Maximum Anti-Squeezing: {np.max(squeezing_dB):.2f} dB")

    return phases, squeezing_dB

# --- Execution Block ---
if __name__ == '__main__':
    # It is highly recommended to disable OpenMP threading when using ProcessPoolExecutor
    # so threads and processes don't fight for CPU time.
    os.environ["OMP_NUM_THREADS"] = "1"
    
    start_time = time.time()
    overlaps_signal, overlaps_vacuum = sim_with_noise_parallel()
    end_time = time.time()
    print(f'Total run time = {(end_time - start_time):.2f} s')



    

#%% Adding in Julia's Plots:
# t = pulse.t_grid
# I_out = np.abs(a_t[-1])**2

# plt.plot(t*1e12, I_out)
# plt.axhline(0, color='k')
# plt.xlabel("Time (ps)")
# plt.ylabel("Intensity")
# plt.title("Output temporal profile")
# plt.show()
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
