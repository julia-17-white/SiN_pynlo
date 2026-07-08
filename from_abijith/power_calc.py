import numpy as np

data = np.load('from_abijith/jw_modes/JW_SiN_AirClad_800nmThickness_1000nmWidth_gamma_aeff.npy')
data_col = np.split(data, 4, axis=1)
gammas = data_col[2]
print(np.mean(gammas))

# parameters:
t_0 = 220e-15 # s (pulse width)
gamma = np.mean(gammas)
N = np.sqrt(2) # N^2 > 2 from Prem Kumar's paper
beta2 = 4.1e-25 # s^2/m from PyNLO simulations
rep_rate = 1e9 # Hz


P0 = (N**2 * beta2) / (gamma * t_0**2)
print(f'power needed = {P0}W')

p_avg = P0 * t_0 * rep_rate
print(f'average power = {p_avg*1000}mW')