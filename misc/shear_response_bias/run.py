import zebu
import numpy as np
import matplotlib.pyplot as plt
from dsigma.physics import critical_surface_density

# %%

table_l = zebu.read_mock_data('lens', 3)
table_s = zebu.read_mock_data('source', 3, survey='des')

# %%

band = 'i'
mag_bins = np.linspace(15, 25, 11)

mag = table_s['mag'][:, table_s.meta['bands'].index(band)]
r = 0.5 * (table_s['R_11'] + table_s['R_22'])

r_ave = []
r_err = []

for i in range(len(mag_bins) - 1):
    use = (mag_bins[i] <= mag) & (mag < mag_bins[i + 1])
    r_ave.append(np.mean(r[use]))
    r_err.append(np.std(r[use]) / np.sqrt(np.sum(use)))

mag = 0.5 * (mag_bins[1:] + mag_bins[:-1])

plt.errorbar(mag, r_ave, yerr=r_err)
plt.xlabel(r'${}$-band magnitude'.format(band))
plt.ylabel(r'response $\langle R_t \rangle$')
plt.tight_layout(pad=0.3)
plt.savefig('response_vs_magn.pdf')
plt.savefig('response_vs_magn.png', dpi=300)

# %%

z_l = np.mean(table_l['z'])
table_s = table_s[table_s['z'] > z_l + 0.2]
r = 0.5 * (table_s['R_11'] + table_s['R_22'])

sigma_crit = critical_surface_density(
    z_l, table_s['z_true'], cosmology=zebu.cosmo, comoving=True)
w_ls = table_s['w'] / sigma_crit**2

shear_1 = (np.average(r / sigma_crit, weights=w_ls) /
           np.average(r, weights=w_ls))
shear_2 = np.average(1.0 / sigma_crit, weights=w_ls)

print('Bias: {:+.1f}%'.format(100 * (shear_1 / shear_2 - 1)))
