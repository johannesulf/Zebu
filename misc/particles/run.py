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
from dsigma.stacking import excess_surface_density


def delta_sigma_from_sigma(rp_bins, sigma, z):

    w = zebu.cosmo.comoving_distance(z).value**-2
    sigma = np.average(sigma, axis=0, weights=w)
    ds = np.zeros_like(sigma)
    w = np.diff(rp_bins**2)
    for i in range(1, len(ds)):
        ds[i] = 0.5 * (np.average(sigma[:i], weights=w[:i]) +
                       np.average(sigma[:i+1], weights=w[:i+1])) - sigma[i]

    ds_mean = np.zeros(len(zebu.rp_bins) - 1)
    for i in range(len(ds_mean)):
        ds_mean[i] = np.average(ds[1+n*i:1+n*(i+1)],
                                weights=w[1+n*i:1+n*(i+1)])

    return ds_mean / 1e12


parser = argparse.ArgumentParser()
parser.add_argument('--compute', action='store_true',
                    help='whether to compute the signal from particles')
args = parser.parse_args()

if args.compute:

    downsample = 1

    ptcl = vstack([Table.read(os.path.join(
        'files', 'downsampled_particles.{}.fits'.format(i))) for i in
        range(64)])[::downsample]
    ra, dec = hp.vec2ang(np.vstack([ptcl['PX'], ptcl['PY'], ptcl['PZ']]).T,
                         lonlat=True)

    ptcl['ra'] = ra
    ptcl['dec'] = dec
    ptcl.keep_columns(['ra', 'dec', 'Z_COS'])
    table_s = dsigma_table(ptcl, 'source', z='Z_COS', w=1, e_1=0, e_2=0,
                           z_l_max=10)

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

        sigma = []

        for i in range(10):
            dz = (zebu.lens_z_bins[lens_bin + 1] -
                  zebu.lens_z_bins[lens_bin]) / 10.0
            z_min = zebu.lens_z_bins[lens_bin] + i * dz
            z_max = z_min + dz
            print('lens redshift: {:.2f} - {:.2f}'.format(z_min, z_max))
            table_l_use = table_l[(table_l['z'] > z_min) &
                                  (table_l['z'] <= z_max)]

            table_s_use = table_s[
                (table_s['z'] > np.amin(table_l_use['z']) - 0.05) &
                (table_s['z'] < np.amax(table_l_use['z']) + 0.05)]

            add_precompute_results(
                table_l_use, table_s_use, rp_bins, cosmology=zebu.cosmo,
                progress_bar=True)

            sigma.append(table_l_use['sum 1'] * 7.82e12 * downsample / (
                np.pi * np.diff(rp_bins**2)))

        sigma = np.concatenate(sigma)

        fname = 'l{}'.format(lens_bin) + '_s{}_gen_zspec_nomag_nofib.hdf5'
        path = os.path.join(zebu.base_dir, 'stacks', 'precompute')
        table_p_l = vstack([Table.read(os.path.join(
            path, fname.format(source_bin)), path='lens') for source_bin in
            range(4)])
        table_p_r = vstack([Table.read(os.path.join(
            path, fname.format(source_bin)), path='random') for source_bin in
            range(4)])
        table_p_l.meta['bins'] = zebu.rp_bins
        table_p_r.meta['bins'] = zebu.rp_bins

        table_l['field_jk'] = hp.ang2pix(
            8, table_l['ra'], table_l['dec'], nest=True, lonlat=True)

        ds_diff = []
        for field_jk in np.unique(table_l['field_jk']):
            use = table_l['field_jk'] != field_jk
            use_p_l = table_p_l['field_jk'] != field_jk
            use_p_r = table_p_r['field_jk'] != field_jk
            ds_diff.append(excess_surface_density(
                    table_p_l[use_p_l], table_p_r[use_p_r],
                    **zebu.stacking_kwargs('gen')) - delta_sigma_from_sigma(
                rp_bins, sigma[use], table_l['z'][use]))

        ds_diff = np.array(ds_diff)

        ds_diff_err = np.sqrt(ds_diff.shape[0] - 1) * np.std(
            ds_diff, ddof=0, axis=0)
        ds_diff_cov = (ds_diff.shape[0] - 1) * np.cov(
            ds_diff, ddof=0, rowvar=False)

        ds_diff = np.mean(ds_diff, axis=0)
        rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])
        ds = delta_sigma_from_sigma(rp_bins, sigma, table_l['z'])

        fname = 'results.csv'

        if os.path.exists(fname):
            table = Table.read(fname)
        else:
            table = Table()
            table['rp'] = rp

        table['ds_{}'.format(lens_bin)] = ds
        table['ds_diff_{}'.format(lens_bin)] = ds_diff
        table['ds_diff_err_{}'.format(lens_bin)] = ds_diff_err

        table.write(fname, overwrite=True)

        fname = 'ds_diff_cov_{}.csv'.format(lens_bin)
        np.savetxt(fname, ds_diff_cov, delimiter=',')

# %%

results = Table.read('results.csv')
rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

color_list = plt.get_cmap('plasma')(
    np.linspace(0.0, 0.8, len(zebu.lens_z_bins) - 1))

for lens_bin in range(len(zebu.lens_z_bins) - 1):
    plt.plot(rp, rp * results['ds_{}'.format(lens_bin)],
             color=color_list[lens_bin], ls='--')
    plt.plot(rp, rp * (results['ds_{}'.format(lens_bin)] +
                       results['ds_diff_{}'.format(lens_bin)]),
             color=color_list[lens_bin])

plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
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
plt.close()

# %%

for lens_bin in range(len(zebu.lens_z_bins) - 1):
    diff = (results['ds_diff_{}'.format(lens_bin)] /
            results['ds_{}'.format(lens_bin)])
    diff_err = (results['ds_diff_err_{}'.format(lens_bin)] /
                results['ds_{}'.format(lens_bin)])
    plotline, caps, barlinecols = plt.errorbar(
        rp * (1 + lens_bin * 0.05), diff, yerr=diff_err, fmt='o',
        color=color_list[lens_bin], ms=2, zorder=lens_bin)
    plt.setp(barlinecols[0], capstyle='round')

plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$\delta \Delta \Sigma / \Delta \Sigma_{\rm ptcl}$')
cmap = mpl.colors.ListedColormap(color_list)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
cb = plt.colorbar(sm, ax=plt.gca(),
                  pad=0.0, ticks=np.linspace(0, 1, len(zebu.lens_z_bins)))
cb.ax.set_yticklabels(['{:g}'.format(z) for z in zebu.lens_z_bins])
cb.ax.minorticks_off()
cb.set_label(r'Lens redshift $z_l$')
plt.axhline(0, color='black', ls='--', zorder=-99)

plt.tight_layout(pad=0.3)
plt.savefig('shear_ptcl_diff.pdf')
plt.savefig('shear_ptcl_diff.png', dpi=300)

# %%

lens_bin = 3
fname = 'ds_diff_cov_{}.csv'.format(lens_bin)
ds_diff_cov = np.genfromtxt(fname, delimiter=',')


def get_mean(y, cov):
    pre = np.linalg.inv(cov)
    w = np.sum(pre, axis=-1)
    return np.average(y, weights=w), np.sqrt(1.0 / np.sum(pre))


ds = results['ds_{}'.format(lens_bin)]
ds_diff = results['ds_diff_{}'.format(lens_bin)] / ds
ds_diff_cov = ds_diff_cov / np.outer(ds, ds)

print(get_mean(ds_diff[-4:], ds_diff_cov[-4:, -4:]))
