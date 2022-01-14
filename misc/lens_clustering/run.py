import os
import zebu
import numpy as np
import multiprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from Corrfunc.mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp

# %%

rp_bins = np.logspace(-1.5, 1.477, 26)
pi_max = 40


def projected_correlation_function(table_l, table_r, rp_bins, pi_max):

    n_l = len(table_l)
    n_r = len(table_r)

    n = multiprocessing.cpu_count()

    ll = DDrppi_mocks(True, 2, n, pi_max, rp_bins, table_l['ra'],
                      table_l['dec'], table_l['d_com'],
                      is_comoving_dist=True)['npairs']
    rr = DDrppi_mocks(True, 2, n, pi_max, rp_bins, table_r['ra'],
                      table_r['dec'], table_r['d_com'],
                      is_comoving_dist=True)['npairs']
    lr = DDrppi_mocks(False, 2, n, pi_max, rp_bins, table_l['ra'],
                      table_l['dec'], table_l['d_com'],
                      RA2=table_r['ra'], DEC2=table_r['dec'],
                      CZ2=table_r['d_com'], is_comoving_dist=True)['npairs']

    wp = convert_rp_pi_counts_to_wp(n_l, n_l, n_r, n_r, ll, lr, lr, rr,
                                    len(rp_bins) - 1, pi_max)

    return wp

# %%


wp_list = []

for i in range(4):

    table_l = zebu.read_mock_data('lens', i)
    table_r = zebu.read_mock_data('random', i)
    table_l['d_com'] = zebu.cosmo.comoving_distance(table_l['z']).value
    table_r['d_com'] = zebu.cosmo.comoving_distance(table_r['z']).value
    wp_list.append(projected_correlation_function(
        table_l, table_r, rp_bins, pi_max))

# %%

rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])


z_bins = zebu.lens_z_bins
color_list = plt.get_cmap('plasma')(np.linspace(0.0, 0.8, len(z_bins) - 1))
cmap = mpl.colors.ListedColormap(color_list)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
cb = plt.colorbar(sm, pad=0.0, ticks=np.linspace(0, 1, len(z_bins)))
cb.ax.set_yticklabels(['{:g}'.format(z) for z in z_bins])
cb.ax.minorticks_off()
cb.set_label(r'Lens redshift $z_l$')

rp_min = 0.2

for offset, wp, color in zip(np.arange(4), wp_list, color_list):
    use = rp > rp_min
    plt.plot(rp[use] * (1 + offset * 0.05), rp[use] * wp[use], color=color)

for i in range(4):
    z_min = zebu.lens_z_bins[i]
    z_max = zebu.lens_z_bins[i + 1]

    if i == 2:
        z_min = 0.6
        z_max = 0.8
    if i == 3:
        z_min = 0.8
        z_max = 1.05

    wp_file = os.path.join(
        'WP', '{}_N_CLUSTERING_wcompEdWsys_z1z2_{:g}-{:g}'.format(
            'BGS_ANY' if i < 2 else 'LRG', z_min, z_max) +
        '_angup-wp-logrp-pi-NJN-120.txt')
    wp_obs = np.genfromtxt(wp_file)[:, 1]
    wp_err = np.genfromtxt(wp_file)[:, 2]
    plotline, caps, barlinecols = plt.errorbar(
        rp[use] * (1 + i * 0.05), rp[use] * wp_obs[use], fmt='.',
        color=color_list[i], yerr=wp_err[use] * rp[use])
    plt.setp(barlinecols[0], capstyle='round')

plt.xscale('log')
plt.xlabel(r'$r_p [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Clustering $r_p \times w_p [h^{-2} \, \mathrm{Mpc}^2]$')
plt.xscale('log')
plt.tight_layout(pad=0.3)
plt.savefig('lens_clustering.pdf')
plt.savefig('lens_clustering.png', dpi=300)
plt.close()
