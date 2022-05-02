import os
import zebu
import argparse
import numpy as np
import healpy as hp
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, vstack
from dsigma.helpers import dsigma_table
from dsigma.jackknife import compress_jackknife_fields
from dsigma.precompute import add_precompute_results
from dsigma.stacking import excess_surface_density

# %%


def delta_sigma_from_sigma(rp_bins, sigma, w, n):

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
        table_r = zebu.read_mock_data('random', lens_bin)[::5]
        table_l['field_jk'] = hp.ang2pix(
            8, table_l['ra'], table_l['dec'], nest=True, lonlat=True)
        table_r['field_jk'] = hp.ang2pix(
            8, table_r['ra'], table_r['dec'], nest=True, lonlat=True)
        z_bins = np.linspace(zebu.lens_z_bins[lens_bin],
                             zebu.lens_z_bins[lens_bin + 1], 100)
        table_l['w_sys'] = interp1d(
            z_bins, zebu.cosmo.comoving_distance(z_bins).value**-2,
            kind='cubic')(table_l['z'])
        table_r['w_sys'] = interp1d(
            z_bins, zebu.cosmo.comoving_distance(z_bins).value**-2,
            kind='cubic')(table_r['z'])

        rp_bins = []
        rp_bins.append([1e-6, zebu.rp_bins[0]])
        n = 10

        for i in range(len(zebu.rp_bins) - 1):
            rp_bins.append(np.linspace(
                zebu.rp_bins[i], zebu.rp_bins[i + 1], n + 1)[1:])

        rp_bins = np.concatenate(rp_bins)

        table_l['sigma'] = np.zeros((len(table_l), len(rp_bins) - 1))
        table_r['sigma'] = np.zeros((len(table_r), len(rp_bins) - 1))

        for i in range(10):
            dz = (zebu.lens_z_bins[lens_bin + 1] -
                  zebu.lens_z_bins[lens_bin]) / 10.0
            z_min = zebu.lens_z_bins[lens_bin] + i * dz
            z_max = z_min + dz
            print('lens redshift: {:.2f} - {:.2f}'.format(z_min, z_max))
            select_l = (table_l['z'] > z_min) & (table_l['z'] <= z_max)
            select_r = (table_r['z'] > z_min) & (table_r['z'] <= z_max)
            select_s = ((table_s['z'] > z_min - 0.1) &
                        (table_s['z'] < z_max + 0.1))

            table_l_select = add_precompute_results(
                table_l[select_l], table_s[select_s], rp_bins,
                cosmology=zebu.cosmo, progress_bar=True, n_jobs=4)
            table_l['sigma'][select_l] = (
                table_l_select['sum 1'] * 7.935e12 * downsample /
                (np.pi * np.diff(rp_bins**2)))

            table_r_select = add_precompute_results(
                table_r[select_r], table_s[select_s], rp_bins,
                cosmology=zebu.cosmo, progress_bar=True, n_jobs=4)
            table_r['sigma'][select_r] = (
                table_r_select['sum 1'] * 7.935e12 * downsample /
                (np.pi * np.diff(rp_bins**2)))

        table_l = compress_jackknife_fields(table_l)
        table_r = compress_jackknife_fields(table_r)

        fname = 'l{}'.format(lens_bin) + '_gen_zspec_nomag_nofib.hdf5'
        fpath = os.path.join(zebu.base_dir, 'stacks', 'precompute', fname)
        table_l_shear = Table.read(fpath, path='lens')
        table_r_shear = Table.read(fpath, path='random')

        ds_diff = []
        for field_jk in np.unique(table_l['field_jk']):
            select = table_l['field_jk'] != field_jk
            ds_particles = delta_sigma_from_sigma(
                rp_bins, table_l['sigma'][select], table_l['w_sys'][select], n)
            select = table_r['field_jk'] != field_jk
            ds_particles -= delta_sigma_from_sigma(
                rp_bins, table_r['sigma'][select], table_r['w_sys'][select], n)
            select_l_shear = table_l_shear['field_jk'] != field_jk
            select_r_shear = table_r_shear['field_jk'] != field_jk
            ds_shear = excess_surface_density(
                table_l_shear[select_l_shear], table_r_shear[select_r_shear],
                random_subtraction=True)
            ds_diff.append(ds_shear - ds_particles)

        ds_diff = np.array(ds_diff)

        ds_diff_err = np.sqrt(ds_diff.shape[0] - 1) * np.std(
            ds_diff, ddof=0, axis=0)
        ds_diff_cov = (ds_diff.shape[0] - 1) * np.cov(
            ds_diff, ddof=0, rowvar=False)

        ds_diff = np.mean(ds_diff, axis=0)
        rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])
        ds = delta_sigma_from_sigma(
            rp_bins, table_l['sigma'], table_l['w_sys'], n)
        ds -= delta_sigma_from_sigma(
            rp_bins, table_r['sigma'], table_r['w_sys'], n)

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
plt.ylabel(r'$(\Delta\Sigma_{\gamma} - \Delta\Sigma_p) / \Delta\Sigma_p$')
cmap = mpl.colors.ListedColormap(color_list)
sm = plt.cm.ScalarMappable(cmap=cmap)
sm._A = []
cb = plt.colorbar(sm, ax=plt.gca(),
                  pad=0.0, ticks=np.linspace(0, 1, len(zebu.lens_z_bins)))
cb.ax.set_yticklabels(['{:g}'.format(z) for z in zebu.lens_z_bins])
cb.ax.minorticks_off()
cb.set_label(r'Lens redshift $z_l$')
plt.axhline(0, color='black', ls='--', zorder=-99)
plt.ylim(-0.25, 0.25)

plt.tight_layout(pad=0.3)
plt.savefig('shear_ptcl_diff.pdf')
plt.savefig('shear_ptcl_diff.png', dpi=300)

# %%

lens_bin = 2
fname = 'ds_diff_cov_{}.csv'.format(lens_bin)
ds_diff_cov = np.genfromtxt(fname, delimiter=',')


def get_mean(y, cov):
    pre = np.linalg.inv(cov)
    w = np.sum(pre, axis=-1)
    return np.average(y, weights=w), np.sqrt(1.0 / np.sum(pre))


ds = results['ds_{}'.format(lens_bin)]
ds_diff = results['ds_diff_{}'.format(lens_bin)] / ds
ds_diff_cov = ds_diff_cov / np.outer(ds, ds)

print(get_mean(ds_diff[-10:], ds_diff_cov[-10:, -10:]))
