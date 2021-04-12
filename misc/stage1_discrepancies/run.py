import os
import zebu
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.spatial import cKDTree
from dsigma.helpers import spherical_to_cartesian
from astropy import units as u
from dsigma.precompute import precompute_photo_z_dilution_factor
from dsigma.precompute import add_maximum_lens_redshift
from dsigma.stacking import photo_z_dilution_factor
from dsigma.physics import critical_surface_density, projection_angle
from scipy.interpolate import interp1d

# %%

lens_bin = 2
source_bin = 3
survey = 'hsc'

table_s = zebu.read_mock_data('source', source_bin, survey=survey)
table_l = zebu.read_mock_data('lens', lens_bin)

# %%

table_c = zebu.read_mock_data('calibration', source_bin, survey=survey)
table_c = add_maximum_lens_redshift(table_c, dz_min=0.2)

z_lens = np.linspace(zebu.lens_z_bins[lens_bin] - 1e-6,
                     zebu.lens_z_bins[lens_bin + 1] + 1e-6, 1000)

weight = np.zeros_like(z_lens)

for i in range(len(z_lens)):
    sigma_crit = critical_surface_density(z_lens[i], table_c['z'],
                                          cosmology=zebu.cosmo)
    weight[i] = np.sum(table_c['w'] * table_c['w_sys'] * sigma_crit**-2 *
                       (z_lens[i] < table_c['z_l_max']))

weight = weight / np.amax(weight)

table_l['w_sys'] = (
    interp1d(z_lens, 1.0 / weight)(table_l['z']) * table_l['w_sys'])

# %%

x_s, y_s, z_s = spherical_to_cartesian(table_s['ra'], table_s['dec'])
kdtree_s = cKDTree(np.column_stack([x_s, y_s, z_s]), leafsize=1000)

# %%

rp_bin = 13

z_s_local = []
z_s_true_local = []
z_l_local = []
w_s_local = []
w_l_local = []

r_min = zebu.rp_bins[rp_bin]
r_max = zebu.rp_bins[rp_bin + 1]

if 'd_com' not in table_l.colnames:
    table_l['d_com'] = zebu.cosmo.comoving_transverse_distance(
        table_l['z']).to(u.Mpc).value

for i, lens in enumerate(table_l):
    
    print(i, len(table_l))

    alpha_min = r_min / lens['d_com']
    alpha_max = r_max / lens['d_com']

    r_3d_min = np.sqrt(2 - 2 * np.cos(alpha_min))
    r_3d_max = np.sqrt(2 - 2 * np.cos(alpha_max))

    x_l, y_l, z_l = spherical_to_cartesian(lens['ra'], lens['dec'])
    sel = np.fromiter(kdtree_s.query_ball_point([x_l, y_l, z_l], r_3d_max),
                      dtype=np.int64)
    r_3d = np.sqrt((x_l - x_s[sel])**2 + (y_l - y_s[sel])**2 +
                   (z_l - z_s[sel])**2)

    sel = sel[r_3d > r_3d_min]

    z_s_local.append(table_s['z'][sel])
    z_s_true_local.append(table_s['z_true'][sel])
    z_l_local.append(np.repeat(lens['z'], len(sel)))
    w_s_local.append(table_s['w'][sel])
    w_l_local.append(np.repeat(lens['w_sys'], len(sel)))

# %%

z_s_local = np.concatenate(z_s_local)
z_s_true_local = np.concatenate(z_s_true_local)
z_l_local = np.concatenate(z_l_local)
w_s_local = np.concatenate(w_s_local)
w_l_local = np.concatenate(w_l_local)

# %%

print(len(w_s_local))

table = Table.read(os.path.join(
    zebu.base_dir, 'stacks', 'region_1', 'precompute',
    'l{}_s{}_{}_nosmag.hdf5'.format(lens_bin, source_bin, survey)))

print(np.sum(table['sum 1'][:, rp_bin]))

# %%

table_c_local = Table()
table_c_local['z'] = z_s_local
table_c_local['z_true'] = z_s_true_local
table_c_local['w'] = w_s_local
table_c_local['w_sys'] = 1.0

# %%

table_c_local = add_maximum_lens_redshift(table_c_local, dz_min=0.2)

# %%

from dsigma.stacking import photo_z_dilution_factor

print(photo_z_dilution_factor(table))

# %%

print(table.colnames)

# %%

print(precompute_photo_z_dilution_factor(0.7, table_c, cosmology=zebu.cosmo))
print(precompute_photo_z_dilution_factor(0.7, table_c_local, cosmology=zebu.cosmo))

# %%

print(table.meta['rp_bins'][rp_bin])

# %%

sigma_crit = critical_surface_density(
    z_l_local, z_s_spec_local, cosmology=zebu.cosmo, comoving=True)
e_t_local_fake = 3.4 / sigma_crit

# %%

use = (z_l_local < z_s_local - 0.2) & (z_l_local > z_s_spec_local - 0.1)
sigma_crit = critical_surface_density(
    z_l_local, z_s_local, cosmology=zebu.cosmo, comoving=True)
ds_phot = (np.sum((w_l_local * w_s_local / sigma_crit * e_t_local)[use]) /
           np.sum((w_l_local * w_s_local / sigma_crit**2)[use]))

# %%

use = (z_l_local < z_s_spec_local - 0.2)
sigma_crit = critical_surface_density(
    z_l_local, z_s_spec_local, cosmology=zebu.cosmo, comoving=True)
ds_spec = (np.sum((w_l_spec_local * w_s_local / sigma_crit * e_t_local)[use]) /
           np.sum((w_l_spec_local * w_s_local / sigma_crit**2)[use]))

# %%

use = (z_l_local < z_s_local - 0.2) & (z_l_local > z_s_spec_local - 0.1)
sigma_crit_phot = critical_surface_density(
    z_l_local, z_s_local, cosmology=zebu.cosmo, comoving=True)
sigma_crit_spec = critical_surface_density(
    z_l_local, z_s_spec_local, cosmology=zebu.cosmo, comoving=True)
f_bias = (np.sum((w_l_local * w_s_local / sigma_crit_phot**2)[use]) /
          np.sum((w_l_local * w_s_local / sigma_crit_phot / sigma_crit_spec)[use]))

# %%

print(ds_phot * f_bias, ds_spec)

# %%

import matplotlib.pyplot as plt

z_l = 0.8
z_s = np.linspace(0.8, 1.5, 100)
plt.plot(z_s, critical_surface_density(z_l, z_s, cosmology=zebu.cosmo,
                                       comoving=False)**-1)

# %%

print(ds_phot, ds_spec)

# %%

bins = np.linspace(0, 2, 51)
plt.hist(z_s_spec_local, bins=bins, histtype='step', density=True,
         label=r'local, $z_{\rm true}$')
plt.hist(table_c['z_true'], bins=bins, histtype='step', density=True,
         label=r'calibration, $z_{\rm true}$')
plt.hist(z_s_local, bins=bins, histtype='step', density=True,
         label=r'local, $z_{\rm phot}$')
plt.hist(table_c['z'], bins=bins, histtype='step', density=True,
         label=r'calibration, $z_{\rm phot}$')
plt.legend(loc='best')

# %%

bins = np.linspace(0.7, 0.9, 21)

use = (z_l_local < z_s_local - 0.2) & (z_l_local < z_s_spec_local - 0.2)
sigma_crit = critical_surface_density(
    z_l_local, z_s_local, cosmology=zebu.cosmo, comoving=True)
weights = w_l_local * w_s_local / sigma_crit**2 * use
plt.hist(z_l_local, weights=weights, density=True, histtype='step', bins=bins)

use = z_l_local < z_s_spec_local - 0.2
sigma_crit = critical_surface_density(
    z_l_local, z_s_spec_local, cosmology=zebu.cosmo, comoving=True)
weights = w_l_spec_local * w_s_local / sigma_crit**2 * use
plt.hist(z_l_local, weights=weights, density=True, histtype='step', bins=bins)

plt.hist(table_l['z'], density=True, histtype='step', bins=bins)