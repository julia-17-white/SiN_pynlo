import numpy as np
import matplotlib.pyplot as plt

data = np.load('from_abijith/Airclad/SiN_airclad_800nm_thick_5000nm_width_neffs.npy')
data_j = np.load('from_abijith/jw_modes/JW_SiN_AirClad_800nmThickness_5000nmWidth_gamma_aeff.npy')
print(data.shape)
print(data_j.shape)
data_col = data[:,1]
# print(data[:,0])
data_j_col = data_j[:,1]
if len(data_col) == len(data_j_col):
    diffs = []
    for i in range(len(data_col)):
        diffs.append(abs(data_col - data_j_col))
else:
    print('data lengths are not equal')

print(f'mean difference = {np.mean(diffs)}')

plt.figure()
plt.plot(data[:,0], data[:,1], label='pre-generated')
plt.plot(data_j[:,0], data_j[:,1], label='i generated')
plt.legend()
plt.xlabel('wavelength')
plt.ylabel('n_eff')
plt.show()