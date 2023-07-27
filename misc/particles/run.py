import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import zebu

from astropy_healpix.healpy import vec2ang, ang2pix
from astropy.table import Table, vstack
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from dsigma.jackknife import compress_jackknife_fields
from dsigma.stacking import excess_surface_density

# %%


def delta_sigma_from_sigma(rp_bins, sigma, n):

    # Let's first estimate DS in each of the fine bins.
    sigma = np.average(sigma, axis=0)
    ds = np.zeros_like(sigma)
    w = np.diff(rp_bins**2)
    for i in range(1, len(ds)):
        ds[i] = 0.5 * (np.average(sigma[:i], weights=w[:i]) +
                       np.average(sigma[:i+1], weights=w[:i+1])) - sigma[i]

    # Now, let's average the results.
    ds_mean = np.zeros((len(rp_bins) - 2) // n)
    for i in range(len(ds_mean)):
        ds_mean[i] = np.average(ds[1+n*i:1+n*(i+1)],
                                weights=w[1+n*i:1+n*(i+1)])

    return ds_mean / 1e12

# %%


parser = argparse.ArgumentParser()
parser.add_argument('--compute', action='store_true',
                    help='whether to compute the signal from particles')
args = parser.parse_args()

if args.compute:

    pixels = zebu.PIXELS
    downsample = 1
    ptcl = vstack([Table.read(os.path.join(
        'files', 'downsampled_particles.{}.fits'.format(i))) for i in
        range(64)])[::downsample]
    ra, dec = vec2ang(np.vstack([ptcl['PX'], ptcl['PY'], ptcl['PZ']]).T,
                      lonlat=True)

    ptcl['ra'] = ra
    ptcl['dec'] = dec
    ptcl.keep_columns(['ra', 'dec', 'Z_COS'])
    table_s_all = dsigma_table(ptcl, 'source', z='Z_COS', w=1, e_1=0, e_2=0)

    for lenses in ['bgs', 'lrg']:
        table_l_all = zebu.read_mock_catalog(
            lenses, zebu.MOCK_PATH / 'buzzard-4', pixels)
        table_l_all['w_sys'] = 1
        table_r_all = zebu.read_mock_catalog(
            lenses + '-r', zebu.MOCK_PATH / 'buzzard-4', pixels)
        table_r_all['w_sys'] = 1
        table_l_all['field_jk'] = ang2pix(
            8, table_l_all['ra'], table_l_all['dec'], nest=True, lonlat=True)
        table_r_all['field_jk'] = ang2pix(
            8, table_r_all['ra'], table_r_all['dec'], nest=True, lonlat=True)
        table_l_all = table_l_all[np.isin(table_l_all['field_jk'], pixels)]
        table_r_all = table_r_all[np.isin(table_r_all['field_jk'], pixels)]
        table_r_all = table_r_all[::3]

        if lenses == 'bgs':
            table_l_all = table_l_all[
                table_l_all['abs_mag_r'] < zebu.ABS_MAG_R_MAX]
            table_l_all = table_l_all[table_l_all['bright'] == 1]
            table_r_all = table_r_all[
                table_r_all['abs_mag_r'] < zebu.ABS_MAG_R_MAX]
            table_r_all = table_r_all[table_r_all['bright'] == 1]

        n = 30
        rp_bins = np.geomspace(zebu.RP_BINS[0], zebu.RP_BINS[-1],
                               (len(zebu.RP_BINS) - 1) * n + 1)
        rp_bins = np.append(rp_bins[::-1], 1e-6)[::-1]

        for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
            for i in range(10):
                dz = (zebu.LENS_Z_BINS[lenses][lens_bin] -
                      zebu.LENS_Z_BINS[lenses][lens_bin + 1]) / 10.0
                z_min = zebu.lens_z_bins[lens_bin] + i * dz
                z_max = z_min + dz
                print('lens redshift: {:.2f} - {:.2f}'.format(z_min, z_max))
                select_l = ((table_l_all['z'] > z_min) &
                            (table_l_all['z'] <= z_max))
                select_r = ((table_r_all['z'] > z_min) &
                            (table_r_all['z'] <= z_max))
                select_s = ((table_s_all['z'] > z_min - 0.075) &
                            (table_s_all['z'] < z_max + 0.075))

                table_l = table_l_all[select_l]
                table_r = table_r_all[select_r]
                table_s = table_s_all[select_s]

                kwargs = dict(cosmology=zebu.COSMOLOGY, progress_bar=True,
                              n_jobs=40, lens_source_cut=None, weighting=0)
                table_l = precompute(table_l, table_s, rp_bins, **kwargs)
                table_r = precompute(table_r, table_s, rp_bins, **kwargs)
                table_l_all['sigma'][select_l] = (
                    table_l['sum 1'] * 7.935e12 * downsample /
                    (np.pi * np.diff(rp_bins**2)))
                table_r_all['sigma'][select_r] = (
                    table_r['sum 1'] * 7.935e12 * downsample /
                    (np.pi * np.diff(rp_bins**2)))

            z_min = zebu.lens_z_bins[lens_bin]
            z_max = zebu.lens_z_bins[lens_bin + 1]
            select_l = ((table_l_all['z'] > z_min) &
                        (table_l_all['z'] <= z_max))
            select_r = ((table_r_all['z'] > z_min) &
                        (table_r_all['z'] <= z_max))
            table_l = table_l_all[select_l]
            table_r = table_r_all[select_r]
            table_l = compress_jackknife_fields(table_l)
            table_r = compress_jackknife_fields(table_r)

            path = zebu.BASE_PATH / 'stacks' / 'results'
            table_l_shear = Table.read(
                path / ('1' if lenses == 'bgs' else '4') /
                'l{}_s3_ds.hdf5'.format(lens_bin))
            table_r_shear = Table.read(
                path / ('7' if lenses == 'bgs' else '10') /
                'l{}_s3_ds.hdf5'.format(lens_bin))
            table_l_shear = table_l_shear[np.isin(
                table_l_shear['field_jk'], pixels)]
            table_r_shear = table_r_shear[np.isin(
                table_r_shear['field_jk'], pixels)]

            table_l.sort('field_jk')
            table_r.sort('field_jk')
            table_l_shear.sort('field_jk')
            table_r_shear.sort('field_jk')

            ds_ptcl = (delta_sigma_from_sigma(rp_bins, table_l['sigma'], n) -
                       delta_sigma_from_sigma(rp_bins, table_r['sigma'], n))
            ds_shear = excess_surface_density(table_l_shear, table_r=table_r)

            ds_ptcl = np.zeros((len(table_l), len(zebu.RP_BINS) - 1))
            ds_shear = np.zeros((len(table_l), len(zebu.RP_BINS) - 1))
            for i in range(len(table_l)):
                select = np.arange(len(table_l)) != i
                ds_ptcl[i, :] = (
                    delta_sigma_from_sigma(
                        rp_bins, table_l['sigma'][select], n) -
                    delta_sigma_from_sigma(
                        rp_bins, table_r['sigma'][select], n))
                ds_shear[i, :] = excess_surface_density(
                    table_l_shear[select], table_r_shear[select],
                    random_subtraction=True)

            table = Table()
            table['ds_ptcl'] = np.mean(ds_ptcl, axis=0)
            table['ds_shear'] = np.mean(ds_shear, axis=0)
            table['ds_diff_err'] = np.sqrt(len(table_l) - 1) * np.std(
                ds_ptcl - ds_shear, ddof=0, axis=0)
            table.write('{}_{}.csv'.format(lenses, lens_bin), overwrite=True)

# %%

for lenses in ['bgs', 'lrg']:
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
        results = Table.read('{}_{}.csv'.format(lenses, lens_bin))
        rp = np.sqrt(zebu.RP_BINS[1:] * zebu.RP_BINS[:-1])
        p = plt.plot(rp, rp * results['ds_ptcl'], ls='-', label='{}-{}'.format(
            lenses.upper(), lens_bin + 1))
        plt.plot(rp, rp * results['ds_shear'], ls='--', color=p[0].get_color())

plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig('pctl_shear.pdf')
plt.savefig('pctl_shear.png', dpi=300)
plt.close()

# %%

offset = 0

for lenses in ['bgs', 'lrg']:
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
        results = Table.read('{}_{}.csv'.format(lenses, lens_bin))
        rp = np.sqrt(zebu.RP_BINS[1:] * zebu.RP_BINS[:-1])
        plotline, caps, barlinecols = plt.errorbar(
            rp * (1 + offset * 0.03), results['ds_ptcl'] / results['ds_shear'],
            yerr=results['ds_diff_err'] / results['ds_shear'], fmt='-o',
            label='{}-{}'.format(lenses.upper(), lens_bin + 1), zorder=offset)
        plt.setp(barlinecols[0], capstyle='round')
        offset += 1

plt.axhline(1.0, ls='--', color='black', zorder=-1)
plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$\Delta \Sigma_{\rm shear} / \Delta \Sigma_{\rm ptcl}$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig('pctl_shear_ratio.pdf')
plt.savefig('pctl_shear_ratio.png', dpi=300)
plt.close()
