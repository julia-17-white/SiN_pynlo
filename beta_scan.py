import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pynlo 
import fiber_sim

from scipy import interpolate, signal
from scipy.constants import c, pi, h
from scipy.optimize import curve_fit


def obtain_data(data_directory):
    '''
    Method to obtain the data for a set in the length scan.
    '''

    all_data = {}
    
    # 1. Get a list of all relevant .csv files in the directory
    try:
        files = [f for f in os.listdir(data_directory) if f.endswith('nmWidth_gamma_aeff.npy')]
    except FileNotFoundError:
        print(f"Directory not found: {data_directory}")
        return all_data

    # 2. Define a helper to extract the float value for sorting
    def extract_width(filename):
        # Remove the prefix from the start of the string
        cleaned = filename.replace('JW_SiN_O2Clad_800nmThickness_', '')
        
        # Remove the suffix from the end of the string
        cleaned = cleaned.replace('nmWidth_gamma_aeff.npy', '')
        
        # Convert the remaining string to a float
        return float(cleaned)

    # 3. Sort the files numerically based on the extracted length
    files.sort(key=extract_width)

    # 4. Loop through the sorted files and read the data
    widths = []
    all_data['beta2_v0'] = []
    all_data['gamma_v0'] = []
    for file_name in files:
        width = extract_width(file_name)
        file_path = os.path.join(data_directory, file_name)
        
        data = np.load(file_path)

        v_min, v_max, v0 = c/4000e-9, c/400e-9, c/1562e-9
        e_p, t_fwhm = 40 * 1e-12, 210e-15
        fib_loss = e_p/(50e-12)
        coup_loss = 1.3
        coup_loss_perc = 10.0 ** (-coup_loss / 10.0)

        n_points = 2**13 # 20 for sidebands
        
        pulse = fiber_sim.nd_run()
        pulse.a_v = pulse.a_v * fib_loss * coup_loss_perc
        # print("Frq Res: {:.3g} GHz".format(pulse.dv * 1e-9))
        v_grid = pulse.v_grid

        # --- 1. Extract and Convert Data
        # Column 0: Wavewidth (assumed microns from modesolver.py)
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
        # alpha_val_per_m = (0.1 / 10.0) * np.log(10) / length
        # alpha_v = np.full_like(v_grid, alpha_val_per_m) # including loss

        mode = pynlo.medium.Mode(v_grid, beta_v, alpha = 0, g3=g3_v)

        beta2 = mode.beta2
        gamma = mode.gamma

        idx_v0 = np.argmin(np.abs(v_grid - v0))
        all_data['beta2_v0'].append(beta2[idx_v0])
        all_data['gamma_v0'].append(gamma[idx_v0])

        
        widths.append(width)

    all_data['widths'] = widths


    return all_data, widths


def plot_gamma(all_data):
    plt.figure()

    plt.scatter(all_data['widths'], all_data['gamma_v0'], s=5)


    plt.grid()
    plt.xlabel('width')
    plt.ylabel('gamma at peak')
    plt.title('gamma plot')
    plt.show()

def plot_beta2(all_data):
    plt.figure()

    plt.scatter(all_data['widths'], all_data['beta2_v0'], s=5)
    plt.axhline(0, label='0', c='k')


    plt.grid()
    plt.xlabel('width')
    plt.ylabel('beta2 at peak')
    plt.title('beta2 plot')
    plt.legend()
    plt.show()

def plot_both(all_data):
    fig, ax1 = plt.subplots()

    ax1.scatter(all_data['widths'], all_data['beta2_v0'], s=8, label=r'$\beta_2$', color='indigo')
    ax1.axhline(0, label=r'$\beta_2 = 0$', color='mediumpurple')
    ax1.set_ylabel(r'$\beta_2$ value at 1560 nm', color='indigo')

    ax2 = ax1.twinx()
    ax2.scatter(all_data['widths'], all_data['gamma_v0'], s=8, label=r'$\gamma$', color='forestgreen')
    ax2.axhline(10.7/1000, label=r'$\gamma$ of HNLF', color='darkseagreen')
    ax2.set_ylabel(r'$\gamma$ value at 1560 nm', color='forestgreen')

    ax1.grid()
    ax1.set_xlabel('Waveguide Width (nm)')
    plt.title(r'$\beta_2$ and $\gamma$ Parameters at Various Waveguide Widths')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    plt.show()


def main():
    file_path1 = 'from_abijith/jw_modes/'

    all_data, widths = obtain_data(file_path1)

    # plot_gamma(all_data)

    # plot_beta2(all_data)

    plot_both(all_data)

if __name__ == '__main__':
    main()