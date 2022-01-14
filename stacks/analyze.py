import os
import zebu
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table, vstack

from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.stacking import boost_factor
from dsigma.jackknife import jackknife_resampling, compress_jackknife_fields

parser = argparse.ArgumentParser()
parser.add_argument('stage', type=int, help='stage of the analysis')
parser.add_argument('--full', action='store_true',
                    help='whether plot everyting')
args = parser.parse_args()

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 3
fiber_assignment = args.stage >= 4
output = 'stage_{}'.format(args.stage)

# %%

if not os.path.isdir(output):
    os.makedirs(output)

if args.stage == 0:
    survey_list = ['gen']
else:
    survey_list = ['des', 'hsc', 'kids']

rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])


def initialize_plot(survey=None, ylabel=None):

    if survey is not None:
        fig, axarr = plt.subplots(figsize=(7, 5), nrows=3, ncols=2,
                                  sharex=True, sharey='row')
    else:
        fig, axarr = plt.subplots(figsize=(7, 2), nrows=1, ncols=3,
                                  sharex=True, sharey='row')

    ax_list = axarr.flatten()
    plt.xscale('log')

    if survey is not None:
        lens_bin_list = []
    else:
        lens_bin_list = np.arange(len(zebu.lens_z_bins) - 1)

    for i, ax in enumerate(ax_list):

        if survey is not None:
            if i <= 3:
                lens_bin = i // 2
            else:
                lens_bin = i - 2
            lens_bin_list.append(lens_bin)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:g}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:+g}\%'.format(y) if y != 0 else r'0\%'))
        ax.axhline(0, ls='--', color='black')

        if survey is not None:
            text = r'${:.1f} \leq z_l < {:.1f}$'.format(
                zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1])
        else:
            text = ['DES', 'HSC', 'KiDS'][i]

        ax.text(0.08, 0.92, text, ha='left', va='top', transform=ax.transAxes,
                zorder=200)

        if (i >= len(ax_list) - 2 and survey is not None):
            ax.set_xlabel(
                r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        if survey is None:
            ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        if (i % 2 == 0 and survey is not None) or (i == 0 and survey is None):
            if ylabel is None:
                ylabel = (r'Residual bias')
            ax.set_ylabel(ylabel)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)

    if survey is not None:
        z_bins = zebu.source_z_bins[survey]
    else:
        z_bins = zebu.lens_z_bins

    color_list = plt.get_cmap('plasma')(np.linspace(0.0, 0.8, len(z_bins) - 1))
    cmap = mpl.colors.ListedColormap(color_list)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    cb = plt.colorbar(sm, ax=ax_list[-1] if survey is None else ax_list,
                      pad=0.0, ticks=np.linspace(0, 1, len(z_bins)))
    cb.ax.set_yticklabels(['{:g}'.format(z) for z in z_bins])
    cb.ax.minorticks_off()

    if survey is not None:
        cb.set_label(r'Photometric source redshift $z_s$')
    else:
        cb.set_label(r'Lens redshift $z_l$')

    if survey is not None:
        source_bin_list = []

        for lens_bin in range(4):
            z_l_max = zebu.lens_z_bins[lens_bin + 1]
            source_bin_use = np.arange(len(zebu.source_z_bins[survey]) - 1)[
                z_l_max < zebu.source_z_bins[survey][1:] - 0.25]

            if lens_bin <= 1:
                n_s = len(source_bin_use)
                source_bin_list.append(source_bin_use[:int(np.ceil(n_s / 2))])
                source_bin_list.append(source_bin_use[int(np.ceil(n_s / 2)):])
            else:
                source_bin_list.append(source_bin_use)
    else:
        source_bin_list = np.repeat('all', 4)

    return fig, ax_list, lens_bin_list, source_bin_list, color_list


def read_precompute(survey, lens_bin, source_bin, zspec=False, noisy=False,
                    lens_magnification=False, source_magnification=False,
                    fiber_assignment=False, iip_weights=True, runit=False):

    table_l_all = Table()
    table_r_all = Table()

    if source_bin == 'all':
        source_bin_all = np.arange(len(zebu.source_z_bins[survey]) - 1)
    else:
        source_bin_all = np.atleast_1d(source_bin)

    for source_bin in source_bin_all:

        path = os.path.join(
            'precompute', 'l{}_s{}_{}'.format(lens_bin, source_bin, survey))

        if noisy:
            path += '_noisy'
        if zspec:
            path += '_zspec'
        if runit:
            path += '_runit'
        if not lens_magnification:
            if not source_magnification:
                path += '_nomag'
            else:
                path += '_nolmag'
        if lens_magnification and not source_magnification:
            raise RuntimeError('Precomputation with lens magnification but ' +
                               'no source magnification has not been ' +
                               'performed.')
        if not fiber_assignment:
            path += '_nofib'
        if fiber_assignment and not iip_weights:
            path += '_noiip'

        if not os.path.isfile(path + '.hdf5'):
            continue

        table_l = Table.read(path + '.hdf5', path='lens')
        table_r = Table.read(path + '.hdf5', path='random')

        table_l_all = vstack([table_l, table_l_all])
        table_r_all = vstack([table_r, table_r_all])
        table_l_all.meta['bins'] = table_l.meta['bins']
        table_r_all.meta['bins'] = table_r.meta['bins']

    if source_bin == 'all':
        table_l_all = compress_jackknife_fields(table_l_all)
        table_r_all = compress_jackknife_fields(table_r_all)

    return table_l_all, table_r_all


def difference(table_l, table_r=None, table_l_2=None, table_r_2=None,
               survey_1=None, survey_2=None, boost=False):

    for survey in [survey_1, survey_2]:
        if survey not in ['gen', 'hsc', 'kids', 'des']:
            raise RuntimeError('Unkown survey!')

    if not boost:
        ds_1 = excess_surface_density(table_l, table_r=table_r,
                                      **zebu.stacking_kwargs(survey_1))
        ds_2 = excess_surface_density(table_l_2, table_r=table_r_2,
                                      **zebu.stacking_kwargs(survey_2))

        return ds_1 - ds_2
    else:
        b_1 = boost_factor(table_l, table_r=table_r)
        b_2 = boost_factor(table_l_2, table_r=table_r_2)
        return b_1 - b_2


def plot_difference(ax, color, table_l_1, table_r_1, table_l_2, table_r_2,
                    survey, survey_2=None, ds_norm=1.0, label=None,
                    offset=0, lens_bin=0, boost=False):

    if len(table_l_1) * len(table_r_1) * len(table_l_2) * len(table_r_2) == 0:
        print('Warning: Received empty result to plot.')
        return None

    if survey_2 is None:
        survey_2 = survey

    dds = difference(
        table_l_1, table_r=table_r_1, table_l_2=table_l_2, table_r_2=table_r_2,
        survey_1=survey, survey_2=survey_2, boost=boost) / ds_norm
    dds_cov = jackknife_resampling(
        difference, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
        table_r_2=table_r_2, survey_1=survey, survey_2=survey_2, boost=boost)
    if hasattr(ds_norm, 'shape'):
        dds_cov = dds_cov / np.outer(ds_norm, ds_norm)
    else:
        dds_cov = dds_cov / ds_norm**2

    if np.all(np.isclose(dds_cov, 0)):
        dds_err = np.zeros(len(np.diag(dds_cov)))
    else:
        dds_err = np.sqrt(np.diag(dds_cov))

    plotline, caps, barlinecols = ax.errorbar(
        rp * (1 + offset * 0.05), 100 * dds, yerr=100 * dds_err, label=label,
        fmt='o', ms=2, color=color, zorder=offset + 100)
    plt.setp(barlinecols[0], capstyle='round')

    return None


def adjust_ylim(ax_list, yabsmin=1.2):

    for ax in ax_list:
        ymin, ymax = ax.get_ylim()
        yabsmax = max(yabsmin, max(np.abs(ymin), np.abs(ymax)))
        ax.set_ylim(-yabsmax, yabsmax)


def savefigs(name):
    plt.savefig(os.path.join(output, name + '.pdf'))
    plt.savefig(os.path.join(output, name + '.png'), dpi=300, transparent=True)

# %%


table_l_ref = []
table_r_ref = []
ds_ref = []

for lens_bin in range(len(zebu.lens_z_bins) - 1):

    table_l, table_r = read_precompute(
        'gen', lens_bin, 'all', zspec=True, lens_magnification=False,
        source_magnification=False, fiber_assignment=False)
    table_l_ref.append(table_l)
    table_r_ref.append(table_r)
    ds_ref.append(excess_surface_density(
        table_l, table_r=table_r, **zebu.stacking_kwargs('gen')))

# %%

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

if args.stage == 0:
    for lens_bin in range(len(zebu.lens_z_bins) - 1):
        plt.plot(rp, rp * ds_ref[lens_bin],
                 label=r'${:.1f} \leq z_l < {:.1f}$'.format(
            zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))

    plt.title("no shape noise, no photo-z's, all sources")
    plt.xscale('log')
    plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.3)
    plt.savefig(os.path.join(output, 'reference.pdf'))
    plt.savefig(os.path.join(output, 'reference.png'), dpi=300)
    plt.close()

# %%

if args.stage == 1 or args.full:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot()

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_survey, table_r_survey = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_survey, table_r_survey,
                            table_l_ref[lens_bin], table_r_ref[lens_bin],
                            survey, survey_2='gen', ds_norm=ds_ref[lens_bin],
                            offset=offset)

    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('reference_comparison')
    plt.close()

# %%

if args.full:
    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey)

        for lens_bin, source_bin_list, ax in zip(
                lens_bin_list, source_bin_list_all, ax_list):
            for offset, source_bin in enumerate(source_bin_list):

                table_l, table_r = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)

                plot_difference(ax, color_list[source_bin], table_l, table_r,
                                table_l_ref[lens_bin], table_r_ref[lens_bin],
                                survey, survey_2='gen',
                                ds_norm=ds_ref[lens_bin], offset=offset)

        adjust_ylim(ax_list)
        savefigs('{}_vs_ref'.format(survey))
        plt.close()

# %%

if args.stage == 1 or args.full:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot()

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_zspec, table_r_zspec = read_precompute(
                survey, lens_bin, 'all', zspec=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_zspec, table_r_zspec, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Photometric redshift dilution')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('redshift_dilution')
    plt.close()

# %%

if args.stage == 1 or args.full:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot()

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_zphot, table_r_zphot = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_zphot['sum w_ls e_t sigma_crit f_bias'] = \
                table_l_zphot['sum w_ls e_t sigma_crit']
            table_r_zphot['sum w_ls e_t sigma_crit f_bias'] = \
                table_r_zphot['sum w_ls e_t sigma_crit']
            table_l_zspec, table_r_zspec = read_precompute(
                survey, lens_bin, 'all', zspec=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_zphot, table_r_zphot,
                            table_l_zspec, table_r_zspec, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Photometric redshift dilution (no correction)')
    plt.ylim(-60, +60)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('redshift_dilution_no_correction')
    plt.close()

# %%

if args.full:
    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey)

        for lens_bin, source_bin_list, ax in zip(
                lens_bin_list, source_bin_list_all, ax_list):
            for offset, source_bin in enumerate(source_bin_list):

                table_l_phot, table_r_phot = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                table_l_spec, table_r_spec = read_precompute(
                    survey, lens_bin, source_bin, zspec=True,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)

                plot_difference(ax, color_list[source_bin], table_l_phot,
                                table_r_phot, table_l_spec, table_r_spec,
                                survey, ds_norm=ds_ref[lens_bin],
                                offset=offset)

        adjust_ylim(ax_list)
        savefigs('{}_phot_vs_spec'.format(survey))
        plt.close()

# %%

if args.stage == 1:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot()

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=True, runit=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_runit, table_r_runit = read_precompute(
                survey, lens_bin, 'all', zspec=True, runit=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_runit, table_r_runit, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Shear response')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('response')
    plt.close()

# %%

if args.stage == 2:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot(ylabel='Bias')

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag, table_r_nomag = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=False,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_nomag, table_r_nomag, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Source magnification')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('source_magnification')
    plt.close()

# %%

if args.stage == 2:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot(ylabel='Bias')

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag, table_r_nomag = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=False,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_nomag, table_r_nomag, survey,
                            offset=offset, boost=True)

    ax_list[1].set_title('Impact of magnification on boost estimator')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('boost_magnification_bias')
    plt.close()

# %%

if args.stage == 2 and args.full:

    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey)

        for lens_bin, source_bin_list, ax in zip(
                lens_bin_list, source_bin_list_all, ax_list):
            for offset, source_bin in enumerate(source_bin_list):

                table_l_mag, table_r_mag = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                table_l_nomag, table_r_nomag = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=False,
                    fiber_assignment=fiber_assignment)

                plot_difference(ax, color_list[source_bin], table_l_mag,
                                table_r_mag, table_l_nomag, table_r_nomag,
                                survey, ds_norm=ds_ref[lens_bin],
                                offset=source_bin)

        adjust_ylim(ax_list)
        savefigs('{}_smag'.format(survey))
        plt.close()

# %%

if args.stage == 3:

    camb_results = zebu.get_camb_results()

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot(ylabel='Bias')

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag, table_r_nomag = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=False,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            ds_lm = lens_magnification_bias(
                table_l_normal, zebu.alpha_l[lens_bin], camb_results,
                photo_z_correction=True)
            ax.plot(rp * (1 + offset * 0.05),
                    100 * ds_lm / ds_ref[lens_bin], color=color, ls='--')

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_nomag, table_r_nomag, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Lens magnification')
    plt.ylim(-5, +50)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('lens_magnification')
    plt.close()

# %%

if args.stage == 3 and args.full:

    camb_results = zebu.get_camb_results()

    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey, r'$\Delta\Sigma_\mathrm{lens \ magn.} / ' +
                            r' \Delta\Sigma_{\rm ref}$')

        for lens_bin, source_bin_list, ax in zip(
                lens_bin_list, source_bin_list_all, ax_list):
            for offset, source_bin in enumerate(source_bin_list):

                table_l_mag, table_r_mag = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)
                table_l_nomag, table_r_nomag = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=False,
                    source_magnification=source_magnification,
                    fiber_assignment=fiber_assignment)

                color = color_list[source_bin]

                ds_lm = lens_magnification_bias(
                    table_l_mag, zebu.alpha_l[lens_bin], camb_results,
                    photo_z_dilution_correction=True)
                ax.plot(rp * (1 + offset * 0.05),
                        100 * ds_lm / ds_ref[lens_bin], color=color, ls='--')

                plot_difference(ax, color, table_l_mag, table_r_mag,
                                table_l_nomag, table_r_nomag, survey,
                                ds_norm=ds_ref[lens_bin], offset=offset)

        savefigs('{}_lmag'.format(survey))
        plt.close()

# %%

if args.stage == 4:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot(ylabel='Bias')

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=True)
            table_l_nofib, table_r_nofib = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=False)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_nofib, table_r_nofib, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Impact of fiber collisions')
    plt.ylim(-7, +7)
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(wspace=0)
    savefigs('fiber_collisions')
    plt.close()

if args.stage == 4:

    fig, ax_list, lens_bin_list, source_bin_list, color_list = \
        initialize_plot(ylabel='Bias')

    for survey, ax in zip(survey_list, ax_list):
        for offset, color, lens_bin in zip(
                np.arange(4), color_list, lens_bin_list):

            table_l_normal, table_r_normal = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=True, iip_weights=False)
            table_l_nofib, table_r_nofib = read_precompute(
                survey, lens_bin, 'all', zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=False)

            plot_difference(ax, color, table_l_normal, table_r_normal,
                            table_l_nofib, table_r_nofib, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    ax_list[1].set_title('Impact of fiber collisions (no correction)')
    plt.ylim(-50, +50)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('fiber_collisions_noiip')
    plt.close()

# %%

if args.stage == 4 and args.full:

    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey)

        for lens_bin, source_bin_list, ax in zip(
                lens_bin_list, source_bin_list_all, ax_list):
            for offset, source_bin in enumerate(source_bin_list):

                table_l_fib, table_r_fib = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=False)
                table_l_nofib, table_r_nofib = read_precompute(
                    survey, lens_bin, source_bin, zspec=False,
                    lens_magnification=lens_magnification,
                    source_magnification=source_magnification,
                    fiber_assignment=True)

                plot_difference(ax, color_list[source_bin], table_l_fib,
                                table_r_fib, table_l_nofib, table_r_nofib,
                                survey, ds_norm=ds_ref[lens_bin],
                                offset=source_bin)

        adjust_ylim(ax_list)
        savefigs('{}_fib'.format(survey))
        plt.close()
