import os
import zebu
import numpy as np
import multiprocessing
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

# %%

theta_bins = np.logspace(np.log10(1), np.log10(100), 21) * u.arcmin


def angular_correlation_function(table_1, table_2, table_r, theta_bins):

    n_1 = len(table_1)
    n_2 = len(table_2)
    n_r = len(table_r)

    n = multiprocessing.cpu_count()

    d1d2 = DDtheta_mocks(
        False, n, theta_bins, table_1['ra'], table_1['dec'], RA2=table_2['ra'],
        DEC2=table_2['dec'])['npairs']
    d1r = DDtheta_mocks(
        False, n, theta_bins, table_1['ra'], table_1['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    d2r = DDtheta_mocks(
        False, n, theta_bins, table_2['ra'], table_2['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    rr = DDtheta_mocks(
        True, 4, theta_bins, table_r['ra'], table_r['dec'])['npairs']

    return convert_3d_counts_to_cf(n_1, n_2, n_r, n_r, d1d2, d1r, d2r, rr)

# %%

w = []
table_s = Table.read(os.path.join(
    zebu.base_dir, 'mocks', 'region_1', 'ra_dec_sample.hdf5'))
z_l = 0.5 * (zebu.lens_z_bins[1:] + zebu.lens_z_bins[:-1])

# %%

for i in range(4):
    theta_bins = np.rad2deg(
        zebu.rp_bins / zebu.cosmo.comoving_distance(z_l[i]).value)
    table_l = zebu.read_mock_data('lens', i)
    table_r = zebu.read_mock_data('random', i)
    if i == 0:
        table_l = table_l[::10]
        table_r = table_r[::10]
    w.append(angular_correlation_function(
        table_s, table_l, table_r, theta_bins))

# %%

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

for i in range(4):
    plt.plot(rp, w[i], label='lens bin {}'.format(i + 1))

plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Angular Clustering $w$')
plt.legend(loc='upper right', frameon=False)
plt.xscale('log')
plt.tight_layout(pad=0.3)
plt.savefig('clustering.pdf')
plt.savefig('clustering.png', dpi=300)
plt.close()
