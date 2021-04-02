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
parser.add_argument('--pdf', action='store_true', help='whether to make PDFs')
args = parser.parse_args()

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 3
fiber_assignment = args.stage >= 4
output = os.path.join('region_{}'.format(args.region),
                      'stage_{}'.format(args.stage))

if args.stage == 0:
    survey_list = ['gen']
else:
    survey_list = ['des', 'hsc', 'kids']

rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])


def read_precompute(survey, lens_bin, source_bin, zspec=False, noisy=False,
                    lens_magnification=False, source_magnification=False,
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
        if not source_magnification:
            if not lens_magnification:
                path += '_nomag'
            else:
                path += '_semimag'
        if lens_magnification and not source_magnification:
            raise RuntimeError('Precomputation with lens magnification but ' +
                               'no source magnification has not been ' +
                               'performed.')
        if not fiber_assignment:
            path += '_nofib'

        if os.path.isfile(path + '.txt'):
            continue

        table_l = Table.read(path + '.hdf5', path='lens')
        table_r = Table.read(path + '.hdf5', path='random')

        table_l_all = vstack([table_l, table_l_all])
        table_r_all = vstack([table_r, table_r_all])
        table_l_all.meta['rp_bins'] = table_l.meta['rp_bins']
        table_r_all.meta['rp_bins'] = table_r.meta['rp_bins']

    return table_l_all, table_r_all

# %%


rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

if args.stage == 0:

    for lens_bin in range(1, len(zebu.lens_z_bins) - 1):

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
    if args.pdf:
        plt.savefig(os.path.join(output, 'reference.pdf'))
    plt.savefig(os.path.join(output, 'reference.png'), dpi=300)
    plt.close()

# %%


def plot_ratio(table_l_1, table_r_1, survey_1, table_l_2, table_r_2, survey_2,
               ds_norm, label, offset, lens_bin):

    if len(table_l_1) == 0 or len(table_l_2) == 0:
        return 1

    dds = zebu.ds_diff(
        table_l_1, table_r=table_r_1, table_l_2=table_l_2, table_r_2=table_r_2,
        survey_1=survey_1, survey_2=survey_2, ds_norm=ds_norm)
    dds_cov = jackknife_resampling(
        zebu.ds_diff, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
        table_r_2=table_r_2, survey_1=survey_1, survey_2=survey_2,
        ds_norm=ds_norm)

    if np.all(np.isclose(dds_cov, 0)):
        return 1

    dds_err = np.sqrt(np.diag(dds_cov))
    dds_ave, dds_ave_cov = zebu.linear_regression(
        rp, dds, dds_cov, return_err=True)
    dds_ave = dds_ave[0]
    dds_ave_err = np.sqrt(dds_ave_cov[0, 0])
    plt.errorbar(
        rp * (1 + offset * 0.03), dds, yerr=dds_err,
        label=r'{}: ${:.3f} \pm {:.3f}$'.format(label, dds_ave, dds_ave_err),
        fmt='.', ms=0)

    return 0

# %%


ds_norm = []

for lens_bin in range(1, len(zebu.lens_z_bins) - 1):

    table_l, table_r = read_precompute(
            'gen', lens_bin, 'all', zspec=True,
            lens_magnification=lens_magnification,
            source_magnification=source_magnification,
            fiber_assignment=fiber_assignment)
    ds_norm.append(excess_surface_density(
        table_l, table_r=table_r, **zebu.stacking_kwargs('gen')))

ds_norm.insert(0, None)


for survey in survey_list:
    for lens_bin in range(1, len(zebu.lens_z_bins) - 1):

        table_l_2, table_r_2 = read_precompute(
            survey, lens_bin, 'all', zspec=True,
            lens_magnification=lens_magnification,
            source_magnification=source_magnification,
            fiber_assignment=fiber_assignment)

        offset = 0
        source_z_bins = zebu.source_z_bins[survey]

        for source_bin in np.arange(len(source_z_bins) - 1):
            try:
                table_l_1, table_r_1 = read_precompute(
                    survey, lens_bin, source_bin, zspec=True,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                label = r'${:.2f} \leq z_s < {:.2f}$'.format(
                    source_z_bins[source_bin], source_z_bins[source_bin+1])
                success = plot_ratio(table_l_1, table_r_1, survey, table_l_2,
                                     table_r_2, survey, ds_norm[lens_bin],
                                     label, offset, lens_bin)
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
        if args.pdf:
            plt.savefig(os.path.join(output, 'l{}_{}_zs.pdf'.format(
                lens_bin, survey)))
        plt.savefig(os.path.join(output, 'l{}_{}_zs.png'.format(
            lens_bin, survey)), dpi=300)
        plt.close()


for survey in survey_list:
    for lens_bin in range(1, len(zebu.lens_z_bins) - 1):

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
                success = plot_ratio(table_l_1, table_r_1, survey, table_l_2,
                                     table_r_2, survey, ds_norm[lens_bin],
                                     label, offset, lens_bin)
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
        if args.pdf:
            plt.savefig(os.path.join(output, 'l{}_{}_phot_vs_spec.pdf'.format(
                lens_bin, survey)))
        plt.savefig(os.path.join(output, 'l{}_{}_phot_vs_spec.png'.format(
            lens_bin, survey)), dpi=300)
        plt.close()

if args.stage == 2:
    for survey in survey_list:
        for lens_bin in range(1, len(zebu.lens_z_bins) - 1):

            offset = 0
            source_z_bins = zebu.source_z_bins[survey]

            for source_bin in np.arange(len(source_z_bins) - 1):
                try:
                    table_l_1, table_r_1 = read_precompute(
                        survey, lens_bin, source_bin, zspec=True,
                        lens_magnification=lens_magnification,
                        source_magnification=True,
                        fiber_assignment=fiber_assignment)
                    table_l_2, table_r_2 = read_precompute(
                        survey, lens_bin, source_bin, zspec=True,
                        lens_magnification=lens_magnification,
                        source_magnification=False,
                        fiber_assignment=fiber_assignment)
                    label = r'${:.2f} \leq z_s < {:.2f}$'.format(
                        source_z_bins[source_bin], source_z_bins[source_bin+1])
                    success = plot_ratio(
                        table_l_1, table_r_1, survey, table_l_2, table_r_2,
                        survey, ds_norm[lens_bin], label, offset, lens_bin)
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
            plt.ylabel(
                r'$(\Delta \Sigma_{\rm smag} - \Delta \Sigma_{\rm no smag})' +
                r'/ \Delta \Sigma_{\rm no smag}$')

            plt.xscale('log')
            plt.axhline(0.0, ls='--', color='black')
            ymin, ymax = plt.gca().get_ylim()
            yabsmax = max(np.abs(ymin), np.abs(ymax))
            plt.ylim(-yabsmax, 2 * yabsmax)
            plt.tight_layout(pad=0.3)
            if args.pdf:
                plt.savefig(os.path.join(
                    output, 'l{}_{}_smag_vs_nosmag.pdf'.format(
                        lens_bin, survey)))
            plt.savefig(os.path.join(
                output, 'l{}_{}_smag_vs_nosmag.png'.format(
                    lens_bin, survey)), dpi=300)
            plt.close()
