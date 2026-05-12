import numpy as np
data = np.load('from_abijith/Effective_Index_Data_Oxide_Cladded/2025_SiN_O2Clad_800nmThickness_900nmWidth.npy')
data_j = np.load('JW_SiN_O2Clad_800nmThickness_900nmWidth.npy')
print(data.shape)
print(data_j.shape)
data_col = np.split(data, 2, axis=1)
data_j_col = np.split(data, 2, axis=1)
if len(data_col[1]) == len(data_j_col[1]):
    diffs = []
    for i in range(len(data_col)):
        diffs.append(abs(data_col[1] - data_j_col[1]))
else:
    print('data lengths are not equal')

print(f'mean difference = {np.mean(diffs)}')