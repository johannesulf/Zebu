import os
import zebu
import argparse
import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from dsigma.helpers import dsigma_table
from dsigma.precompute import add_precompute_results

# %%

parser = argparse.ArgumentParser()
parser.add_argument('--compute', action='store_true',
                    help='whether to compute the signal from particles')
args = parser.parse_args()

if args.compute:

    downsample = 1

    ptcl = vstack([Table.read(os.path.join(
        'files', 'downsampled_particles.{}.fits'.format(i))) for i in
        range(64)])[::downsample]
    ptcl = ptcl[ptcl['Z_COS'] > 0.6]
    ra, dec = hp.vec2ang(np.vstack([ptcl['PX'], ptcl['PY'], ptcl['PZ']]).T,
                         lonlat=True)

    ptcl['ra'] = ra
    ptcl['dec'] = dec
    ptcl.keep_columns(['ra', 'dec'])
    table_s = dsigma_table(ptcl, 'source', z=10, w=1, e_1=0, e_2=0, z_l_max=10)

    for lens_bin in range(len(zebu.lens_z_bins) - 1):
        table_l = zebu.read_mock_data(
            'lens', lens_bin, unlensed_coordinates=True)

        rp_bins = []
        rp_bins.append([1e-6, zebu.rp_bins[0]])
        n = 10

        for i in range(len(zebu.rp_bins) - 1):
            rp_bins.append(np.linspace(
                zebu.rp_bins[i], zebu.rp_bins[i + 1], n + 1)[1:])

        rp_bins = np.concatenate(rp_bins)

        add_precompute_results(table_l, table_s, rp_bins, cosmology=zebu.cosmo,
                               progress_bar=True)

        rp = np.sqrt(rp_bins[1:] * rp_bins[:-1])
        w = zebu.cosmo.comoving_distance(table_l['z']).value**-2
        sigma = np.average(
            table_l['sum 1'], axis=0, weights=w) * 7.82e12 * downsample
        sigma /= np.pi * np.diff(rp_bins**2)
        ds = np.zeros_like(sigma)
        w = np.diff(rp_bins**2)
        for i in range(1, len(ds)):
            ds[i] = 0.5 * (np.average(sigma[:i], weights=w[:i]) +
                           np.average(sigma[:i+1], weights=w[:i+1])) - sigma[i]

        ds_mean = np.zeros(len(zebu.rp_bins) - 1)
        for i in range(len(ds_mean)):
            ds_mean[i] = np.average(ds[1+n*i:1+n*(i+1)],
                                    weights=w[1+n*i:1+n*(i+1)])

        rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

        fname = 'ptcl.csv'

        if os.path.exists(fname):
            table = Table.read(fname)
        else:
            table = Table()
            table['rp'] = rp

        table['ds_{}'.format(lens_bin)] = ds_mean / 1e12

        table.write(fname, overwrite=True)

# %%

ptcl = Table.read('ptcl.csv')
shear = Table.read(os.path.join(
    zebu.base_dir, 'stacks', 'stage_0', 'shear.csv'))
rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

color_list = plt.get_cmap('plasma')(
    np.linspace(0.0, 0.8, len(zebu.lens_z_bins) - 1))

for lens_bin in range(len(zebu.lens_z_bins) - 1):
    plt.plot(rp, rp * ptcl['ds_{}'.format(lens_bin)],
             color=color_list[lens_bin])
    plt.plot(rp, rp * shear['ds_{}'.format(lens_bin)],
             color=color_list[lens_bin], ls='--')

plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
cmap = mpl.colors.ListedColormap(color_list)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
cb = plt.colorbar(sm, ax=plt.gca(),
                  pad=0.0, ticks=np.linspace(0, 1, len(zebu.lens_z_bins)))
cb.ax.set_yticklabels(['{:g}'.format(z) for z in zebu.lens_z_bins])
cb.ax.minorticks_off()
cb.set_label(r'Lens redshift $z_l$')

plt.tight_layout(pad=0.3)
plt.savefig('pctl_shear.pdf')
plt.savefig('pctl_shear.png', dpi=300)
