import numpy as np
from astropy.table import Table
from make_mocks import read_buzzard_catalog, is_BGS, is_LRG

# %%

pixel_list = [340, 341, 395, 396, 398, 399, 416, 417, 418, 419, 420, 421, 422,
              424, 425, 426, 637, 638, 639]
mu_list = [0.90, 0.95, 1.00, 1.05, 1.10]

# %%

table_m = []
z_bins = [0.1, 0.3, 0.5, 0.7, 0.9]

for i in range(4):
    table_m.append(Table())
    table_m[-1].meta['mu'] = mu_list
    table_m[-1]['pixel'] = pixel_list
    table_m[-1]['n'] = np.zeros((len(pixel_list), len(mu_list)))


for i, pixel in enumerate(pixel_list):
    table_b = read_buzzard_catalog(pixel)
    table_b.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
    for j, mu in enumerate(mu_list):
        table_b_magn = table_b.copy()
        table_b_magn['mag'] += -2.5 * np.log10(mu)
        for k in range(4):
            use_z = ((z_bins[k] <= table_b_magn['z_true']) &
                     (table_b_magn['z_true'] < z_bins[k + 1]))
            if k <= 1:
                table_m[k]['n'][i, j] = np.sum(is_BGS(table_b_magn) & use_z)
            else:
                table_m[k]['n'][i, j] = np.sum(is_LRG(table_b_magn) & use_z)

# %%

for i in range(4):
    if i == 0:
        table_m[i].write('magnification.hdf5', path='lens_0', overwrite=True,
                         serialize_meta=True)
    else:
        table_m[i].write('magnification.hdf5', path='lens_{}'.format(i),
                         append=True, serialize_meta=True)
