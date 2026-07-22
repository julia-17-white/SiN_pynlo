import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, pi
import sys
sys.path.append("/Users/juliaspc/Desktop/cu_boulder/monte_carlo_all/PyNLO")
import pynlo
import copy  # Added for cloning the pulse
import pandas as pd

def setup_waveguide_and_pulse():
    # --- Parameters ---
    FWHM    = 0.205   # pulse duration (ps)
    pulseWL = 1562.5    # pulse central wavelength (nm)
    EPP     = 50e-12  # Energy per pulse (J)
    Window  = 20.0    # simulation window (ps)
    Points  = 2**13  

    # Dispersion values (ps^2/km) 
    beta2_pm = -22.0*10**-27 # PM-1550 (Anomalous)
    beta2_nd = 17.5035*10**-27  # ND Fiber (Normal) -> from Tsung Han
    beta3_val = 1e-40

    n_points = 2**13 # 20 for sidebands

    # --- 1. Pulse Creation ---
    v0    = c / (pulseWL * 1e-9)
    dv    = 1.0 / (Window * 1e-12)
    v_min = v0 - (Points/2) * dv
    v_max = v0 + (Points/2 - 1) * dv

    # Standard positional arguments for your version
    pulse = pynlo.light.Pulse.Sech(Points, v_min, v_max, v0, EPP, FWHM * 1e-12)
    v_grid = pulse.v_grid

    # --- 2. Construct Constant Nonlinearity ---
    gamma_pm = 0.00012  # Example value for PM-1550
    gamma_nd = 0.0045  # Example value for ND Fiber

    gamma_array_pm = np.full_like(v_grid, gamma_pm)
    g3_v_pm = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma_array_pm)
    gamma_array_nd = np.full_like(v_grid, gamma_nd)
    g3_v_nd = pynlo.utility.chi3.gamma_to_g3(v_grid, gamma_array_nd)

    # --- 3. Construct Analytical Dispersion ---
    # Build beta(omega) using a Taylor expansion around the center frequency.
    # We set beta_0 = 0 and beta_1 = 0 since we operate in the co-moving frame.
    w_grid = 2 * np.pi * v_grid
    w0 = 2 * np.pi * v0
    dw = (w_grid - w0)

    beta_v_pm = 0.5 * beta2_pm * dw**2 + ((1.0/6.0)*beta3_val*dw**3)
    beta_v_nd = 0.5 * beta2_nd * dw**2 + ((1.0/6.0)*beta3_val*dw**3)

    dt = pulse.dt

    # --- 4. Setup pynlo Mode ---
    mode_pm = pynlo.medium.Mode(v_grid, beta_v_pm, alpha=None, g3=g3_v_pm)
    mode_nd = pynlo.medium.Mode(v_grid, beta_v_nd, alpha=None, g3=g3_v_nd)

    return pulse, mode_nd, mode_pm, v_grid

def propagate_pulse(pulse, mode, length):
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
    # dz = dz = 1e-3  # 1 mm step size guarantees numerical stability
    # dz = length / 2000   # 2000 steps over 5 mm → 2.5 µm steps #JULIA REDEFINED THIS

    new_pulse, z, a_t, a_v = sim.simulate(length, dz=dz, local_error=local_error, n_records=100, plot=None)
    # change the plot to "frq" if you want it to plot --> I don't want it to do that plot 100+ times so I set plot=None

    return new_pulse, z, a_t, a_v, sim


def plot_osa_spectrum(a_v, sim, pulse, pulse_coh):
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
    p_in_per_nm = np.abs(pulse_coh.a_v)**2 * sim.dv_dl * 1e-9
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
    
    # 4. Create the Plot
    plt.figure("OSA Spectrum", figsize=(9, 6))
    
    plt.plot(wvl_nm, p_in_dB, color="tab:blue", label="Input")
    plt.plot(wvl_nm, p_out_dB, color="tab:green", label="Output")
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Intensity (dB/nm)")
    plt.title("Simulated OSA Output Spectrum")
    
    # Adjust these limits based on your specific pulse bandwidth
    # plt.xlim(1000, 2200) 
    plt.xlim(1500, 1620)
    
    # Dynamically scale the y-axis to focus on the top 60 dB of the signal
    # plt.ylim(np.max(p_out_dB) - 60, np.max(p_out_dB) + 5)
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    file = 'laser.CSV'
    data = pd.read_csv(file, skiprows = 44, header=None, names=['nm', 'dB/nm'])
    data['dB_norm'] = data['dB/nm'] - data['dB/nm'].max()

    plt.figure()
    plt.plot(wvl_nm, p_out_dB, color="indigo", label="PyNLO")
    plt.plot(data['nm'], data['dB_norm'], color='darkorange', label='Measured')
    plt.ylim(-80,5)
    plt.xlim(1500, 1625)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Relative Intensity (dB/nm)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.title('Pulse Spectrum into the Waveguide', fontsize=18)
    plt.show()

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
    cb_ax = fig.add_axes([.91,.125,.02,.454])
    fig.colorbar(im,orientation='vertical',cax=cb_ax)

    ax0.set_ylabel('Intensity (dB)')
    ax2.set_ylabel('Length (mm)', labelpad = 20)
    plt.show()

def nd_run():
    pulse, mode_nd, mode_pm, v_grid = setup_waveguide_and_pulse()
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(pulse, mode_pm, length=1.69)
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(new_pulse, mode_nd, length=1.73)

    return new_pulse

if __name__ == '__main__':
    pulse, mode_nd, mode_pm, v_grid = setup_waveguide_and_pulse()
    pulse_init = copy.deepcopy(pulse)
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(pulse, mode_pm, length=1.7)
    # nice_plot(a_v_out, sim, new_pulse, a_t, z)
    new_pulse, z, a_t, a_v_out, sim = propagate_pulse(new_pulse, mode_nd, length=1.7)
    # nice_plot(a_v_out, sim, new_pulse, a_t, z)
    plot_osa_spectrum(a_v_out, sim, new_pulse, pulse_init)

