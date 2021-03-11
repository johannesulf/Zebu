import os
import zebu
import argparse
import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import jackknife_resampling

parser = argparse.ArgumentParser()
parser.add_argument('stage', type=int, help='stage of the analysis')
parser.add_argument('--region', type=int, help='region of the sky', default=1)
args = parser.parse_args(['0', ])

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 0
fiber_assignment = args.stage >= 3
output = os.path.join('region_{}'.format(args.region),
                      'stage_{}'.format(args.stage))

if args.stage == 0:
    survey_list = ['gen']
else:
    if args.source_bin < 4:
        survey_list = ['des', 'hsc', 'kids']
    else:
        survey_list = ['kids']

rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])


def read_precompute(survey, lens_bin, source_bin, zspec=False, noisy=False,
                    lens_magnification=True, source_magnification=True,
                    fiber_assignment=False):

    table_l_all = Table()
    table_r_all = Table()

    if source_bin == 'all':
        source_bin_all = np.arange(len(zebu.source_z_bins[survey]) - 1)
    else:
        source_bin_all = np.atleast_1d(source_bin)

    for source_bin in source_bin_all:

        path = os.path.join('region_{}'.format(args.region), 'precompute',
                            'l{}_s{}_{}'.format(lens_bin, source_bin, survey))

        if noisy:
            path += '_noisy'
        if zspec:
            path += '_zspec'
        if fiber_assignment:
            path += '_fiber'
        if not source_magnification:
            path += '_nosmag'
        if not lens_magnification:
            path += '_nolmag'

        if os.path.isfile(path + '.hdf5'):

            table_l = Table.read(path + '.hdf5', path='lens')
            table_r = Table.read(path + '.hdf5', path='random')

            # The photo-z calibration factors aren't weighted by the number
            # of sources in each source redshift bin. This can be a problem if
            # we combine results from all source bins. So ensure calibration
            # here.
            norm_c = (np.sum(table_l['sum w_ls']) /
                      np.sum(table_l['calib: sum w_ls w_c']))
            table_l['calib: sum w_ls w_c'] *= norm_c
            table_l['calib: sum w_ls w_c ' +
                    'sigma_crit_p / sigma_crit_t'] *= norm_c
            table_r['calib: sum w_ls w_c'] *= norm_c
            table_r['calib: sum w_ls w_c ' +
                    'sigma_crit_p / sigma_crit_t'] *= norm_c

            table_l_all = vstack([table_l, table_l_all])
            table_r_all = vstack([table_r, table_r_all])
            table_l_all.meta['rp_bins'] = table_l.meta['rp_bins']
            table_r_all.meta['rp_bins'] = table_r.meta['rp_bins']

        elif os.path.isfile(path + '.txt'):
            continue

        else:
            print(path)
            raise FileNotFoundError()

    return table_l_all, table_r_all

# %%


rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

if args.stage == 0:

    for lens_bin in range(2, len(zebu.lens_z_bins) - 1):

        table_l, table_r = read_precompute(
            'gen', lens_bin, 'all', zspec=True,
            lens_magnification=lens_magnification,
            source_magnification=source_magnification,
            fiber_assignment=fiber_assignment)

        kwargs = zebu.stacking_kwargs(survey='gen')
        kwargs['table_r'] = table_r
        ds = excess_surface_density(table_l, **kwargs)

        plt.plot(rp, rp * ds, label=r'${:.1f} \leq z_l < {:.1f}$'.format(
            zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))

    plt.title("no shape noise, no photo-z's, all sources")
    plt.xscale('log')
    plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(output, 'reference.pdf'))
    plt.savefig(os.path.join(output, 'reference.png'), dpi=300)
    plt.close()

# %%


def plot_ratio(table_l_1, table_r_1, survey_1, table_l_2, table_r_2, survey_2,
               ds_norm, label, offset):

    if len(table_l_1) == 0 or len(table_l_2) == 0:
        return 1

    dds = zebu.ds_diff(
        table_l_1, table_r=table_r_1, table_l_2=table_l_2, table_r_2=table_r_2,
        survey_1=survey_1, survey_2=survey_2, ds_norm=ds_norm, stage=1)
    dds_cov = jackknife_resampling(
        zebu.ds_diff, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
        table_r_2=table_r_2, survey_1=survey_1, survey_2=survey_2,
        ds_norm=ds_norm, stage=1)

    if np.all(np.isclose(dds_cov, 0)):
        return 1

    dds_err = np.sqrt(np.diag(dds_cov))
    i_min = np.arange(len(rp))[rp > 0.5][0]
    dds_ave, dds_ave_cov = zebu.linear_regression(
        rp[i_min:], dds[i_min:], dds_cov[i_min:, i_min:], return_err=True)
    dds_ave = dds_ave[0]
    dds_ave_err = np.sqrt(dds_ave_cov[0, 0])
    plt.errorbar(
        rp[i_min:] * (1 + offset * 0.03), dds[i_min:], yerr=dds_err[i_min:],
        label=r'{}: ${:.3f} \pm {:.3f}$'.format(label, dds_ave, dds_ave_err),
        fmt='.', ms=0)

    return 0

# %%


if args.stage == 0:
    for lens_bin in range(2, len(zebu.lens_z_bins) - 1):

        table_l_2, table_r_2 = read_precompute(
            'gen', lens_bin, 'all', zspec=True,
            lens_magnification=lens_magnification,
            source_magnification=source_magnification,
            fiber_assignment=fiber_assignment)
        ds_norm = excess_surface_density(
            table_l_2, table_r=table_r_2, **zebu.stacking_kwargs('gen'))

        offset = 0
        source_z_bins = zebu.source_z_bins['gen']

        for source_bin in np.arange(len(source_z_bins) - 1):
            try:
                table_l_1, table_r_1 = read_precompute(
                    'gen', lens_bin, source_bin, zspec=True,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                label = r'${:.2f} \leq z_s < {:.2f}$'.format(
                        source_z_bins[source_bin], source_z_bins[source_bin+1])
                success = plot_ratio(table_l_1, table_r_1, 'gen', table_l_2,
                                     table_r_2, 'gen', ds_norm, label, offset)
                if success == 0:
                    offset = offset + 1
            except FileNotFoundError:
                pass

        if offset == 0:
            continue

        plt.title(r"spec-z's, ${:.1f} < z_l < {:.1f}$".format(
            zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))
        plt.legend(loc='upper center', frameon=False)
        plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        plt.ylabel(r'$(\Delta \Sigma - \Delta \Sigma_{\rm all}) /' +
                   r'\Delta \Sigma_{\rm all}$')

        plt.xscale('log')
        plt.axhline(0.0, ls='--', color='black')
        ymin, ymax = plt.gca().get_ylim()
        yabsmax = max(np.abs(ymin), np.abs(ymax))
        plt.ylim(-yabsmax, 2 * yabsmax)
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(output, 'reference_l{}_zs.pdf'.format(
            lens_bin)))
        plt.savefig(os.path.join(output, 'reference_l{}_zs.png'.format(
            lens_bin)), dpi=300)
        plt.close()


for survey in survey_list:
    for lens_bin in range(2, len(zebu.lens_z_bins) - 1):

        offset = 0
        source_z_bins = zebu.source_z_bins[survey]

        for source_bin in np.arange(len(source_z_bins) - 1):
            try:
                table_l_1, table_r_1 = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                table_l_2, table_r_2 = read_precompute(
                    survey, lens_bin, source_bin, zspec=True,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                label = r'${:.2f} \leq z_s < {:.2f}$'.format(
                        source_z_bins[source_bin], source_z_bins[source_bin+1])
                success = plot_ratio(table_l_1, table_r_1, 'gen', table_l_2,
                                     table_r_2, 'gen', ds_norm, label, offset)
                if success == 0:
                    offset = offset + 1
            except FileNotFoundError:
                pass

        if offset == 0:
            continue

        plt.title(r"${:.1f} < z_l < {:.1f}$".format(
            zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))
        plt.legend(loc='upper center', frameon=False)
        plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        plt.ylabel(r'$(\Delta \Sigma_{\rm phot} - \Delta \Sigma_{\rm spec})' +
                   r'/ \Delta \Sigma_{\rm spec}$')

        plt.xscale('log')
        plt.axhline(0.0, ls='--', color='black')
        ymin, ymax = plt.gca().get_ylim()
        yabsmax = max(np.abs(ymin), np.abs(ymax))
        plt.ylim(-yabsmax, 2 * yabsmax)
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(output, 'l{}_phot_vs_spec.pdf'.format(
            lens_bin)))
        plt.savefig(os.path.join(output, 'l{}_phot_vs_spec.png'.format(
            lens_bin)), dpi=300)
        plt.close()
