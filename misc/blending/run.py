import os
import zebu
import numpy as np
from astropy.table import Table
import matplotlib as mpl
import matplotlib.pyplot as plt
from dsigma.precompute import add_precompute_results
from dsigma.physics import critical_surface_density

# %%

w = []
table_s = Table.read(os.path.join(
    zebu.base_dir, 'mocks', 'mocks', 'i_band_sample.hdf5'))
table_s['z'] = 10.0
table_s['z_l_max'] = 10.0
table_s['e_1'] = 0.0
table_s['e_2'] = 0.0
table_s = table_s[table_s['w'] < 100]

for key in table_s.colnames:
    table_s[key] = table_s[key].astype(float)

w = []

for i in range(len(zebu.lens_z_bins) - 1):
    table_l = zebu.read_mock_data('lens', i)
    table_r = zebu.read_mock_data('random', i)
    if i == 0:
        table_l = table_l[::10]
        table_r = table_r[::10]
    if i < 2:
        table_s_use = table_s[::10]
    else:
        table_s_use = table_s
    table_l['w_sys'] = critical_surface_density(
        table_l['z'], 10.0, cosmology=zebu.cosmo)**2
    table_r['w_sys'] = critical_surface_density(
        table_r['z'], 10.0, cosmology=zebu.cosmo)**2
    add_precompute_results(
        table_l, table_s_use, zebu.rp_bins, cosmology=zebu.cosmo,
        progress_bar=True)
    add_precompute_results(
        table_r, table_s_use, zebu.rp_bins, cosmology=zebu.cosmo,
        progress_bar=True)
    w.append(np.mean(table_l['sum w_ls'].data *
                     table_l['w_sys'].data[:, None], axis=0) /
             np.mean(table_r['sum w_ls'].data *
                     table_r['w_sys'].data[:, None], axis=0) - 1)

# %%

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])
z_bins = zebu.lens_z_bins
color_list = plt.get_cmap('plasma')(np.linspace(0.0, 0.8, len(z_bins) - 1))
cmap = mpl.colors.ListedColormap(color_list)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
cb = plt.colorbar(sm, pad=0.0, ticks=np.linspace(0, 1, len(z_bins)))
cb.ax.set_yticklabels(['{:g}'.format(z) for z in z_bins])
cb.ax.minorticks_off()
cb.set_label(r'Lens redshift $z_l$')

for i, color in enumerate(color_list):
    plt.plot(rp, w[i], color=color)

plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Angular clustering $w_i$')
plt.xscale('log')
plt.tight_layout(pad=0.8)
plt.savefig('clustering.pdf')
plt.savefig('clustering.png', dpi=300)
plt.close()
