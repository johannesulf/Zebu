import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import zebu

from astropy.table import Table, vstack
from astropy import units as u
from astropy_healpix.healpy import vec2ang, ang2pix
from dsigma.helpers import dsigma_table
from dsigma.physics import critical_surface_density
from dsigma.precompute import precompute
from dsigma.jackknife import compress_jackknife_fields
from dsigma.stacking import excess_surface_density
from scipy.interpolate import interp1d


DOWNSAMPLE = 1
PARTICLE_MASS = 7.94069e12 * DOWNSAMPLE


# %%

def lens_weight(z_l, z_s):
    # Determine the effective weight assigned to each lens galaxy in the
    # galaxy-galaxy lensing measurements.
    return np.mean([
        zebu.COSMOLOGY.comoving_distance(z_l).value**-2 *
        critical_surface_density(z_l, z, cosmology=zebu.COSMOLOGY)**-2 *
        (z_l + 0.2 < z) for z in z_s])


def delta_sigma_from_sigma(rp_bins, sigma, weights, n):

    # Let's first estimate DS in each of the fine bins.
    sigma = np.average(sigma, weights=weights, axis=0)
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
    ptcl = vstack([Table.read(os.path.join(
        'files', 'downsampled_particles.{}.fits'.format(i))) for i in
        range(64)])[::DOWNSAMPLE]
    ra, dec = vec2ang(np.vstack([ptcl['PX'], ptcl['PY'], ptcl['PZ']]).T,
                      lonlat=True)

    ptcl['ra'] = ra
    ptcl['dec'] = dec
    ptcl.keep_columns(['ra', 'dec', 'Z_COS'])
    table_s_all = dsigma_table(ptcl, 'source', z='Z_COS', w=1, e_1=0, e_2=0)

    table_s = zebu.read_mock_catalog('hsc', zebu.MOCK_PATH / 'buzzard-4',
                                     zebu.PIXELS)
    table_s = table_s[
        np.digitize(table_s['z'], zebu.SOURCE_Z_BINS['hsc']) == 4]
    z_s = np.random.choice(table_s['z_true'], 1000)

    for lenses in ['bgs', 'lrg']:
        table_l_all, table_r_all = zebu.read_mock_catalog(
            [lenses, lenses + '-r'], zebu.MOCK_PATH / 'buzzard-4', pixels,
            magnification=False, unlensed_coordinates=True)
        table_l_all = table_l_all[
            (table_l_all['z'] >= zebu.LENS_Z_BINS[lenses][0]) &
            (table_l_all['z'] <= zebu.LENS_Z_BINS[lenses][-1])]
        table_r_all = table_r_all[
            (table_r_all['z'] >= zebu.LENS_Z_BINS[lenses][0]) &
            (table_r_all['z'] <= zebu.LENS_Z_BINS[lenses][-1])]
        table_l_all['field_jk'] = ang2pix(
            8, table_l_all['ra'], table_l_all['dec'], nest=True, lonlat=True)
        table_r_all['field_jk'] = ang2pix(
            8, table_r_all['ra'], table_r_all['dec'], nest=True, lonlat=True)
        table_l_all = table_l_all[np.isin(table_l_all['field_jk'], pixels)]
        table_r_all = table_r_all[np.isin(table_r_all['field_jk'], pixels)]
        table_r_all = table_r_all[::2]

        # Account for the fact that the GGL estimator weighs lenses differently
        # depending on redshift.
        z_l = np.linspace(zebu.LENS_Z_BINS[lenses][0],
                          zebu.LENS_Z_BINS[lenses][-1], 1000)
        table_l_all['w_sys'] = interp1d(
            z_l, [lens_weight(z, z_s) for z in z_l],
            kind='cubic')(table_l_all['z'])
        table_r_all['w_sys'] = interp1d(
            z_l, [lens_weight(z, z_s) for z in z_l],
            kind='cubic')(table_r_all['z'])

        if lenses == 'bgs':
            table_l_all = table_l_all[table_l_all['bright'] == 1]
            table_r_all = table_r_all[table_r_all['bright'] == 1]
            for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
                table_l_all = table_l_all[np.where(
                    np.digitize(table_l_all['z'], zebu.LENS_Z_BINS['bgs'])
                    != lens_bin + 1, True, table_l_all['abs_mag_r'] <
                    zebu.ABS_MAG_R_MAX[lens_bin])]
                table_r_all = table_r_all[np.where(
                    np.digitize(table_r_all['z'], zebu.LENS_Z_BINS['bgs'])
                    != lens_bin + 1, True, table_r_all['abs_mag_r'] <
                    zebu.ABS_MAG_R_MAX[lens_bin])]

        n = 25
        rp_bins = np.geomspace(zebu.RP_BINS[0], zebu.RP_BINS[-1],
                               (len(zebu.RP_BINS) - 1) * n + 1)
        rp_bins = np.append(rp_bins[::-1], 1e-6)[::-1]
        table_l_all['sigma'] = np.zeros((len(table_l_all), len(rp_bins) - 1))
        table_r_all['sigma'] = np.zeros((len(table_r_all), len(rp_bins) - 1))

        for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
            for i in range(10):
                dz = (zebu.LENS_Z_BINS[lenses][lens_bin + 1] -
                      zebu.LENS_Z_BINS[lenses][lens_bin]) / 10.0
                z_min = zebu.LENS_Z_BINS[lenses][lens_bin] + i * dz
                z_max = z_min + dz
                print('lens redshift: {:.2f} - {:.2f}'.format(z_min, z_max))
                select_l = ((table_l_all['z'] > z_min) &
                            (table_l_all['z'] <= z_max))
                select_r = ((table_r_all['z'] > z_min) &
                            (table_r_all['z'] <= z_max))
                select_s = ((table_s_all['z'] > z_min - 0.15) &
                            (table_s_all['z'] < z_max + 0.15))

                table_l = table_l_all[select_l]
                table_r = table_r_all[select_r]
                table_s = table_s_all[select_s]
                d_l = zebu.COSMOLOGY.comoving_distance(
                    0.5 * (z_min + z_max)).to(u.Mpc).value

                kwargs = dict(cosmology=zebu.COSMOLOGY, progress_bar=True,
                              n_jobs=40, lens_source_cut=None, weighting=0)
                table_l = precompute(table_l, table_s, rp_bins, **kwargs)
                table_r = precompute(table_r, table_s, rp_bins, **kwargs)
                area = - 2 * np.pi * d_l**2 * np.diff(np.cos(rp_bins / d_l))
                table_l_all['sigma'][select_l] = (
                    table_l['sum 1'] * PARTICLE_MASS / area)
                table_r_all['sigma'][select_r] = (
                    table_r['sum 1'] * PARTICLE_MASS / area)

            z_min = zebu.LENS_Z_BINS[lenses][lens_bin]
            z_max = zebu.LENS_Z_BINS[lenses][lens_bin + 1]
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
                path / ('0' if lenses == 'bgs' else '1') /
                'l{}_s3_ds.hdf5'.format(lens_bin))
            table_r_shear = Table.read(
                path / ('2' if lenses == 'bgs' else '3') /
                'l{}_s3_ds.hdf5'.format(lens_bin))
            table_l_shear = table_l_shear[np.isin(
                table_l_shear['field_jk'], pixels)]
            table_r_shear = table_r_shear[np.isin(
                table_r_shear['field_jk'], pixels)]

            table_l.sort('field_jk')
            table_r.sort('field_jk')
            table_l_shear.sort('field_jk')
            table_r_shear.sort('field_jk')

            ds_ptcl = (delta_sigma_from_sigma(
                rp_bins, table_l['sigma'], table_l['w_sys'], n) -
                delta_sigma_from_sigma(
                rp_bins, table_r['sigma'], table_r['w_sys'], n))
            ds_shear = excess_surface_density(table_l_shear, table_r=table_r)

            ds_ptcl = np.zeros((len(table_l), len(zebu.RP_BINS) - 1))
            ds_shear = np.zeros((len(table_l), len(zebu.RP_BINS) - 1))
            for i in range(len(table_l)):
                select = np.arange(len(table_l)) != i
                ds_ptcl[i, :] = (
                    delta_sigma_from_sigma(
                        rp_bins, table_l['sigma'][select],
                        table_l['w_sys'][select], n) -
                    delta_sigma_from_sigma(
                        rp_bins, table_r['sigma'][select],
                        table_r['w_sys'][select], n))
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

# Use the corrections for reduced shear.
table = Table.read(zebu.BASE_PATH / 'stacks' / 'plots_relative' /
                   'reduced_shear_ds_hsc.csv')
table = table[table['source_bin'] == 3]
c = np.split((table['value'].data / 100) + 1, 6)


for lenses in ['bgs', 'lrg']:
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
        results = Table.read('{}_{}.csv'.format(lenses, lens_bin))
        results['ds_shear'] /= c[lens_bin + 3 * (lenses == 'lrg')]
        rp = np.sqrt(zebu.RP_BINS[1:] * zebu.RP_BINS[:-1])
        color = mpl.colormaps['plasma'](
            (lens_bin + (lenses == 'lrg') * 3 + 0.5) / 5.0)
        plt.plot(rp, rp * results['ds_ptcl'], ls='-', label='{}-{}'.format(
            lenses.upper(), lens_bin + 1), color=color)
        plt.plot(rp, rp * results['ds_shear'], ls='--', color=color)

plt.xscale('log')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig('ptcl_shear.pdf')
plt.savefig('ptcl_shear.png', dpi=300)
plt.close()

# %%

fig, axarr = plt.subplots(nrows=2, sharex=True, figsize=(3.33, 3.33))

for ax, lenses in zip(axarr, ['bgs', 'lrg']):
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
        results = Table.read('{}_{}.csv'.format(lenses, lens_bin))[:-2]
        results['ds_shear'] /= c[lens_bin + 3 * (lenses == 'lrg')][:-2]
        rp = np.sqrt(zebu.RP_BINS[1:] * zebu.RP_BINS[:-1])[:-2]
        color = mpl.colormaps['plasma'](
            (lens_bin + (lenses == 'lrg') * 3) / 5.0)
        x = rp * (1 + lens_bin * 0.03)
        y = results['ds_shear'] / results['ds_ptcl']
        y_err = results['ds_diff_err'] / results['ds_shear']
        ax.plot(x, y, label='{}-{}'.format(lenses.upper(), lens_bin + 1),
                zorder=lens_bin, color=color)
        plotline, caps, barlinecols = ax.errorbar(
            x, y, yerr=y_err, fmt='o', zorder=lens_bin, color=color)
        plt.setp(barlinecols[0], capstyle='round')

for ax in axarr:
    ax.axhline(1.0, ls='--', color='black', zorder=-1)
    ax.set_ylabel(r'$\Delta \Sigma_{\rm shear} / \Delta \Sigma_{\rm ptcl}$')
    ax.legend(loc='best', frameon=False)

axarr[0].set_ylim(0.78, 1.12)
axarr[1].set_ylim(0.955, 1.065)
axarr[1].set_yticks(np.arange(0.97, 1.07, 0.03))

plt.xscale('log')
axarr[1].set_xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.tight_layout(pad=0.3)
plt.subplots_adjust(hspace=0)
plt.savefig('ptcl_shear_ratio.pdf')
plt.savefig('ptcl_shear_ratio.png', dpi=300)
plt.close()
