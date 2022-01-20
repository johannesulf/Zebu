import numpy as np
from tqdm import tqdm
from astropy.table import Table
from make_mocks import read_buzzard_catalog, is_BGS, is_LRG

# %%

pixel_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
              23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39,
              48, 49, 50, 51, 52, 53, 54, 55, 69, 70, 71, 73, 74, 75, 76, 77,
              78]

# %%

table_m = Table()
z_bins = [0.1, 0.3, 0.5, 0.7, 0.9]
table_m['pixel'] = pixel_list

for i in range(4):
    table_m['alpha_{}'.format(i)] = np.zeros(len(pixel_list))

for i, pixel in enumerate(tqdm(pixel_list)):

    table_mag = read_buzzard_catalog(pixel, mag_lensed=True)
    table_nomag = read_buzzard_catalog(pixel, mag_lensed=False)
    table_mag.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
    table_nomag.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']

    for j in range(4):

        use = table_mag['z_true'] > z_bins[j]
        use &= table_mag['z_true'] < z_bins[j + 1]
        use &= np.abs(table_mag['mu'] - 1) < 0.1

        if j < 2:
            target_mag = is_BGS(table_mag)
            target_nomag = is_BGS(table_nomag)
        else:
            target_mag = is_LRG(table_mag)
            target_nomag = is_LRG(table_nomag)

        x = table_mag['mu'][use] - 1
        y = target_mag[use].astype(int) - target_nomag[use].astype(int)
        m = np.sum(x * y) / np.sum(x**2)
        table_m['alpha_{}'.format(j)][i] = m / np.mean(target_nomag[use])

# %%

table_m.write('magnification.hdf5', overwrite=True, path='data')
