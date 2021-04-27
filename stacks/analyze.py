import os
import zebu
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table, vstack

from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.jackknife import jackknife_resampling, compress_jackknife_fields

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

if not os.path.isdir(output):
    os.makedirs(output)

if args.stage == 0:
    survey_list = ['gen']
else:
    survey_list = ['des', 'hsc', 'kids']

rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])


def initialize_plot(survey, ylabel=None):

    fig, axarr = plt.subplots(figsize=(7, 5), nrows=3, ncols=2, sharex=True,
                              sharey='row')
    ax_list = axarr.flatten()
    plt.xscale('log')

    lens_bin_list = []

    for i, ax in enumerate(ax_list):
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
        ax.text(0.08, 0.92, r'${:.1f} \leq z_l < {:.1f}$'.format(
            zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]),
                ha='left', va='top', transform=ax.transAxes,
                bbox=dict(facecolor='white', edgecolor='black'))
        if i >= 2:
            ax.set_xlabel(
                r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        if i % 2 == 0:
            if ylabel is None:
                ylabel = (r'$\Delta\Sigma_\mathrm{' + survey +
                          r'\ vs. \ ref} / \Delta\Sigma_{\rm ref}$')
            ax.set_ylabel(ylabel)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)

    color_list = plt.get_cmap('plasma')(np.linspace(
        0.0, 0.9, len(zebu.source_z_bins[survey]) - 1))
    cmap = mpl.colors.ListedColormap(color_list)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    cb = plt.colorbar(sm, ax=ax_list, pad=0.0,
                      ticks=np.linspace(0, 1, len(zebu.source_z_bins[survey])))
    cb.ax.set_yticklabels(
        ['{:g}'.format(z_s) for z_s in zebu.source_z_bins[survey]])
    cb.set_label(r'Photometric source redshift $z_s$')

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

    return fig, ax_list, lens_bin_list, source_bin_list, color_list


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

        if os.path.isfile(path + '.txt'):
            continue

        table_l = Table.read(path + '.hdf5', path='lens')
        table_r = Table.read(path + '.hdf5', path='random')

        table_l_all = vstack([table_l, table_l_all])
        table_r_all = vstack([table_r, table_r_all])
        table_l_all.meta['rp_bins'] = table_l.meta['rp_bins']
        table_r_all.meta['rp_bins'] = table_r.meta['rp_bins']

    if source_bin == 'all':
        table_l_all = compress_jackknife_fields(table_l_all)
        table_r_all = compress_jackknife_fields(table_r_all)

    return table_l_all, table_r_all


def difference(table_l, table_r=None, table_l_2=None, table_r_2=None,
               survey_1=None, survey_2=None):

    for survey in [survey_1, survey_2]:
        if survey not in ['gen', 'hsc', 'kids', 'des']:
            raise RuntimeError('Unkown survey!')

    ds_1 = excess_surface_density(table_l, table_r=table_r,
                                  **zebu.stacking_kwargs(survey_1))
    ds_2 = excess_surface_density(table_l_2, table_r=table_r_2,
                                  **zebu.stacking_kwargs(survey_2))

    if survey_1 == 'hsc':
        ds_1 = ds_1 * 2
    if survey_2 == 'hsc':
        ds_2 = ds_2 * 2

    return ds_1 - ds_2


def polyfit(x, y, cov, deg=1):

    X = np.zeros((0, len(x)))
    for i in range(deg + 1):
        X = np.vstack((X, x**i))
    X = X.T

    pre = np.linalg.inv(cov)
    beta_cov = np.linalg.inv(np.dot(np.dot(X.T, pre), X))
    beta = np.dot(np.dot(np.dot(beta_cov, X.T),  pre), y)

    return beta, beta_cov


def polyval(x, beta, beta_cov):

    A = np.array([x**i for i in range(len(beta))])

    return np.sum(A * beta), np.sqrt(np.dot(np.dot(A.T, beta_cov),  A))


def plot_difference(ax, color, table_l_1, table_r_1, table_l_2, table_r_2,
                    survey, survey_2=None, ds_norm=1.0, label=None,
                    offset=0, lens_bin=0):

    if len(table_l_1) * len(table_r_1) * len(table_l_2) * len(table_r_2) == 0:
        print('Warning: Received empty result to plot.')
        return None

    if survey_2 is None:
        survey_2 = survey

    dds = difference(
        table_l_1, table_r=table_r_1, table_l_2=table_l_2, table_r_2=table_r_2,
        survey_1=survey, survey_2=survey_2) / ds_norm
    dds_cov = jackknife_resampling(
        difference, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
        table_r_2=table_r_2, survey_1=survey, survey_2=survey_2)
    if hasattr(ds_norm, 'shape'):
        dds_cov = dds_cov / np.outer(ds_norm, ds_norm)
    else:
        dds_cov = dds_cov / ds_norm**2

    if np.all(np.isclose(dds_cov, 0)):
        raise RuntimeError('Tried to plot differences for identical results.')

    dds_err = np.sqrt(np.diag(dds_cov))
    plotline, caps, barlinecols = ax.errorbar(
        rp * (1 + offset * 0.05), 100 * dds, yerr=100 * dds_err, label=label,
        fmt='.', ms=0, color=color, zorder=offset + 100)
    plt.setp(barlinecols[0], capstyle='round')

    beta, beta_cov = polyfit(np.log10(rp), dds, dds_cov, deg=5)
    rp_mod = np.logspace(np.amin(np.log10(zebu.rp_bins)),
                         np.amax(np.log10(zebu.rp_bins)), 100)
    dds_mod = np.zeros(len(rp_mod))
    dds_mod_err = np.zeros(len(rp_mod))

    for i in range(len(rp_mod)):
        dds_mod[i], dds_mod_err[i] = polyval(np.log10(rp_mod[i]), beta,
                                             beta_cov)
    ax.plot(rp_mod * (1 + offset * 0.05), 100 * dds_mod, color=color,
            zorder=offset + 50, lw=0.5)
    ax.fill_between(
        rp_mod * (1 + offset * 0.05), 100 * (dds_mod - dds_mod_err),
        100 * (dds_mod + dds_mod_err), color=color, alpha=0.3, lw=0,
        zorder=offset)

    return None


def adjust_ylim(ax_list, yabsmin=1.2):

    for ax in ax_list:
        ymin, ymax = ax.get_ylim()
        yabsmax = max(yabsmin, max(np.abs(ymin), np.abs(ymax)))
        ax.set_ylim(-yabsmax, yabsmax)


def savefigs(name):
    if args.pdf:
        plt.savefig(os.path.join(output, name + '.pdf'))
    plt.savefig(os.path.join(output, name + '.png'), dpi=300)

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
    plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
    plt.legend(loc='upper left')
    plt.tight_layout(pad=0.3)
    if args.pdf:
        plt.savefig(os.path.join(output, 'reference.pdf'))
    plt.savefig(os.path.join(output, 'reference.png'), dpi=300)
    plt.close()

# %%

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
                            survey, survey_2='gen', ds_norm=ds_ref[lens_bin],
                            offset=offset)

    adjust_ylim(ax_list)
    savefigs('{}_vs_ref'.format(survey))
    plt.close()

# %%

for survey in survey_list:

    fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
        initialize_plot(survey, r'$\Delta\Sigma_\mathrm{phot \ vs. \ spec} /' +
                        r' \Delta\Sigma_{\rm ref}$')

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
                            table_r_phot, table_l_spec, table_r_spec, survey,
                            ds_norm=ds_ref[lens_bin], offset=offset)

    adjust_ylim(ax_list)
    savefigs('{}_phot_vs_spec'.format(survey))
    plt.close()

# %%

if args.stage == 2:

    for survey in survey_list:

        fig, ax_list, lens_bin_list, source_bin_list_all, color_list = \
            initialize_plot(survey, r'$\Delta\Sigma_\mathrm{source \ magn.} ' +
                            r'/ \Delta\Sigma_{\rm ref}$')

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
