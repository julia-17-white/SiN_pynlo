import numpy as np
data = np.load('from_abijith/Airclad/SiN_airclad_800nm_thick_1000nm_width_neffs.npy')
data_j = np.load('from_abijith/jw_modes/JW_SiN_AirClad_800nmThickness_1000nmWidth.npy')
print(data.shape)
# print(data_j.shape)
data_col = np.split(data, 2, axis=1)
# print(data_col)
data_j_col = np.split(data, 2, axis=1)
if len(data_col[1]) == len(data_j_col[1]):
    diffs = []
    for i in range(len(data_col)):
        diffs.append(abs(data_col[1] - data_j_col[1]))
else:
    print('data lengths are not equal')

print(f'mean difference = {np.mean(diffs)}')