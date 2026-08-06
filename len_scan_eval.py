import os, copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def n_square_m(all_data, lengths):
    '''
    Method to plot the squared soliton number vs the number of soliton periods
    and color by phi_NL.
    '''
    plt.figure()
    levels = [0.3, 1, 3, 10, 30]

    all_squeezing = []
    all_n_squared = []
    
    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['dB squeezing'][i] < 0:
                if all_data[length]['beta_2 at peak'][i] < 0:
                    all_squeezing.append(all_data[length]['dB squeezing'][i])
                    all_n_squared.append(all_data[length]['N^2'][i])
        
    vmin = min(all_squeezing)
    vmax = max(all_squeezing)


    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['dB squeezing'][i] < 0:
                if all_data[length]['beta_2 at peak'][i] < 0:
                    sc = plt.scatter(
                        all_data[length]['N^2'][i], 
                        all_data[length]['number of soliton periods'][i], 
                        c=all_data[length]['dB squeezing'][i], 
                        cmap='viridis',      # You can change this to any matplotlib colormap (e.g., 'plasma', 'inferno')
                        vmin=vmin, 
                        vmax=vmax,
                        label=f"{length} m"
                    )

    cbar = plt.colorbar(sc)
    cbar.set_label('dB Squeezing')

    # 4. Add lines for constant phi_NL
    # Define an array of x-values spanning your data's range
    x_min = min(all_n_squared)
    x_max = max(all_n_squared)
    x_vals = np.logspace(np.log10(x_min), np.log10(x_max), 100)

    for phi in levels:
        # Assuming standard relationship: m = (2 * phi_NL) / (pi * N^2)
        # Modify this equation if your normalization definitions differ!
        y_vals = (2 * phi) / (np.pi * x_vals) 
        
        plt.plot(x_vals, y_vals, linestyle='--', color='gray', alpha=0.5)
        
        # Add a text label to the end of each line for clarity
        # plt.text(x_vals[-1], y_vals[-1], f'$\phi_{{NL}}={phi}$', fontsize=9, color='gray')


    # plt.legend()
    plt.ylabel('Number of Soliton Periods')
    plt.xlabel(r'$N^2 = L_D/L_{NL}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Waveguides in Soliton Parameter Space')

    plt.show()


def meas_sqz(all_data, lengths, anti_sqz):
    '''
    Method to plot the measured amount of squeezing or anti-squeezing vs the width and length of 
    the waveguide. This will have options over whether or not to subtract off
    the amount of shot noise change from interference.
    '''
    if anti_sqz == True:
        data_val = 'dB anti-squeezing'
    else:
        data_val = 'dB squeezing'

    plt.figure()

    all_squeezing = []
    
    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['dB squeezing'][i] < 0:
                if all_data[length]['beta_2 at peak'][i] < 0:
        # if np.min(all_data[length]['dB squeezing']) < 0:
                    all_squeezing.append(all_data[length][data_val][i])

                # if length == .03:
                #     print(all_data[length]['dB constructive interference'][i])

        # print(all_squeezing)
        
    vmin = min(all_squeezing)
    vmax = max(all_squeezing)


    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['dB squeezing'][i] < 0:
                if all_data[length]['beta_2 at peak'][i] < 0:
        # if np.min(all_data[length]['dB squeezing']) < 0:
                    sc = plt.scatter(
                        length, 
                        all_data[length]['width'][i], 
                        c=all_data[length][data_val][i], 
                        cmap='viridis',      # You can change this to any matplotlib colormap (e.g., 'plasma', 'inferno')
                        vmin=vmin, 
                        vmax=vmax,
                        label=f"{length} m"
                    )


    cbar = plt.colorbar(sc)
    cbar.set_label(data_val)
    plt.xscale('log')

    plt.ylabel('Waveguide Width (nm)')
    plt.xlabel('Waveguide Length (m)')
    plt.title(f'Simulated {data_val} for various waveguides')

    plt.show()


def beta2_plot(all_data, lengths):
    '''
    Method to plot the value of beta_2 (as a color) vs the waveguide width and length.
    '''
    plt.figure()

    all_beta2 = []
    
    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['beta_2 at peak'][i] < 0:
                all_beta2.append(all_data[length]['beta_2 at peak'][i])
        
    vmin = min(all_beta2)
    vmax = max(all_beta2)


    for length in lengths:
        for i in range(len(all_data[length]['width'])):
            if all_data[length]['beta_2 at peak'][i] < 0:
                sc = plt.scatter(
                    length, 
                    all_data[length]['width'][i], 
                    c=all_data[length]['beta_2 at peak'][i], 
                    cmap='viridis',      # You can change this to any matplotlib colormap (e.g., 'plasma', 'inferno')
                    vmin=vmin, 
                    vmax=vmax,
                    label=f"{length} m"
                )

    cbar = plt.colorbar(sc)
    cbar.set_label('Beta2 Value')
    plt.xscale('log')

    plt.ylabel('Waveguide Width (nm)')
    plt.xlabel('Waveguide Length (m)')
    plt.title(r'Simulated $\beta_2$ for various waveguides')

    plt.show()



def gamma_plot(all_data, lengths):
    '''
    Method to plot the value of gamma (as a color) vs the waveguide width and length.
    '''
    plt.figure()

    all_gamma = []
    
    for length in lengths:
        all_gamma.extend(all_data[length]['gamma at peak'])
        
    vmin = min(all_gamma)
    vmax = max(all_gamma)


    for length in lengths:
        sc = plt.scatter(
            np.ones(len(all_data[length]['width'])) * length, 
            all_data[length]['width'], 
            c=all_data[length]['gamma at peak'], 
            cmap='viridis',      # You can change this to any matplotlib colormap (e.g., 'plasma', 'inferno')
            vmin=vmin, 
            vmax=vmax,
            label=f"{length} m"
        )

    cbar = plt.colorbar(sc)
    cbar.set_label('Gamma Value')
    plt.xscale('log')

    plt.ylabel('Waveguide Width (nm)')
    plt.xlabel('Waveguide Length (m)')
    plt.title(r'Simulated $\gamma$ for various waveguides')

    plt.show()
    

def obtain_data(data_directory):
    '''
    Method to obtain the data for a set in the length scan.
    '''

    all_data = {}
    
    # 1. Get a list of all relevant .csv files in the directory
    try:
        files = [f for f in os.listdir(data_directory) if f.endswith('m.csv')]
    except FileNotFoundError:
        print(f"Directory not found: {data_directory}")
        return all_data

    # 2. Define a helper to extract the float value for sorting
    def extract_length(filename):
        # Remove 'm.csv' and convert the remaining string to a float
        return float(filename.replace('m.csv', ''))

    # 3. Sort the files numerically based on the extracted length
    files.sort(key=extract_length)

    # 4. Loop through the sorted files and read the data
    lengths = []
    for file_name in files:
        length = extract_length(file_name)
        file_path = os.path.join(data_directory, file_name)
        
        # NOTE: If your CSVs have actual column headers, remove `header=None`. 
        # Using header=None assigns integer headers (0, 1, 2...). 
        df = pd.read_csv(file_path)
        df = df.iloc[:, 1:]
        
        # Optional: If you literally want the column headers themselves to be sorted alphabetically/numerically
        df = df.reindex(sorted(df.columns), axis=1)

        if length != 0.1:
            df = df.rename(columns={
                'dB anti-squeezing': 'dB squeezing', 
                'dB squeezing': 'dB anti-squeezing'
            }) # I messed up two of the solumn names in saving the files. This fixes it without my having to modify all of the .csv files
            # I fixed it when I re-ran the 10cm waveguides lol

        # df.to_dict(orient='list') automatically groups by column headers
        all_data[length] = df.to_dict(orient='list')
        lengths.append(length)

    return all_data, lengths



def main():
    data_directory = 'wvgd_outputs/length_scan/'

    all_data, lengths = obtain_data(data_directory)
    # print("Loaded data for lengths:", list(all_data.keys()))
    # print(all_data[.003].keys())

    # print(all_data[.01]['beta_2 at peak'])

    n_square_m(all_data, lengths)

    meas_sqz(all_data, lengths, False)

    beta2_plot(all_data, lengths)

    gamma_plot(all_data, lengths)



if __name__ == '__main__':
    main()
