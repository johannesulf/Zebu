import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zebu

from astropy.table import Table
from astropy.io.ascii import convert_numpy
from dsigma.jackknife import jackknife_resampling
from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.stacking import tangential_shear
from matplotlib import gridspec
from pathlib import Path


def read_precomputed_data(
        lenses, sources, statistic, lens_magnification=False,
        source_magnification=False, fiber_assignment=False,
        intrinsic_alignment=False, photometric_redshifts=True,
        shear_bias=False, shape_noise=False):

    if lenses in ['bgs-r', 'lrg-r']:
        source_magnification = False
        lens_magnification = False
        intrinsic_alignment = False

    converters = {'*': [convert_numpy(typ) for typ in (int, float, bool, str)]}
    table = Table.read('config.csv', converters=converters)
    select = np.ones(len(table), dtype=bool)
    select &= table['lenses'] == lenses.lower()
    select &= table['sources'] == sources.lower()
    select &= table['lens magnification'] == lens_magnification
    select &= table['source magnification'] == source_magnification
    select &= table['fiber assignment'] == fiber_assignment
    select &= table['intrinsic alignment'] == intrinsic_alignment
    select &= table['photometric redshifts'] == photometric_redshifts
    select &= table['shear bias'] == shear_bias
    select &= table['shape noise'] == shape_noise
    if np.sum(select) == 0:
        raise ValueError('Configuration with these options not available.')

    config = table['configuration'][select][0]
    path = Path('results', '{}'.format(config))

    data = []
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses.split('-')[0]]) - 1):
        data.append([])
        for source_bin in range(len(zebu.SOURCE_Z_BINS[sources]) - 1):
            fname = 'l{}_s{}_{}.hdf5'.format(lens_bin, source_bin, statistic)
            table = Table.read(path / fname)
            n_pairs = np.sum(table['sum 1'], axis=1)
            table = table[n_pairs > 0.01 * np.amax(n_pairs)]
            data[-1].append(table)

    return data


def difference(table_l, table_l_2=None, table_r=None, table_r_2=None,
               function=None, stacking_kwargs=None):

    return (function(table_l_2, table_r=table_r_2, **stacking_kwargs) -
            function(table_l, table_r=table_r, **stacking_kwargs))


def plot_results(file_stem, statistic='ds', config={}, title=None,
                 plot_lens_magnification=False):

    config = dict(
        lens_magnification=False, source_magnification=False,
        fiber_assignment=False, intrinsic_alignment=False,
        photometric_redshifts=True, shear_bias=False,
        shape_noise=False) | config
    config['statistic'] = statistic

    kwargs_1 = {}
    kwargs_2 = {}
    for key in config.keys():
        if isinstance(config[key], tuple):
            kwargs_1[key] = config[key][0]
            kwargs_2[key] = config[key][1]
        else:
            kwargs_1[key] = config[key]
            kwargs_2[key] = config[key]

    survey_list = ['des', 'des', 'kids']
    table_l_1 = [read_precomputed_data('bgs', survey, **kwargs_1) +
                 read_precomputed_data('lrg', survey, **kwargs_1) for
                 survey in survey_list]
    table_r_1 = [read_precomputed_data('bgs-r', survey, **kwargs_1) +
                 read_precomputed_data('lrg-r', survey, **kwargs_1) for
                 survey in survey_list]
    table_l_2 = [read_precomputed_data('bgs', survey, **kwargs_2) +
                 read_precomputed_data('lrg', survey, **kwargs_2) for
                 survey in survey_list]
    table_r_2 = [read_precomputed_data('bgs-r', survey, **kwargs_2) +
                 read_precomputed_data('lrg-r', survey, **kwargs_2) for
                 survey in survey_list]

    n_s = len(table_l_1)
    n_bins_l = len(table_l_1[0])

    fig = plt.figure(figsize=(7, 5))
    gs = gridspec.GridSpec(n_s, n_bins_l + 1, wspace=0,
                           width_ratios=[20] * n_bins_l + [1])
    axes = []
    colors = []

    if statistic == 'ds':
        x = 0.5 * (zebu.RP_BINS[1:] + zebu.RP_BINS[:-1])
    else:
        x = 0.5 * (zebu.THETA_BINS[1:] + zebu.THETA_BINS[:-1])

    for i, survey in enumerate(survey_list):
        axes.append([])
        for j in range(n_bins_l):
            axes[-1].append(fig.add_subplot(
                gs[i, j], sharex=axes[i][-1] if j > 0 else None,
                sharey=axes[i][-1] if j > 0 else None))
            ax = axes[-1][-1]
            ax.axhline(0, ls='--', color='black')
            n_bgs = len(zebu.LENS_Z_BINS['bgs']) - 1
            text = '{}-{}'.format('BGS' if j < n_bgs else 'LRG', j + 1 -
                                  n_bgs * (j >= n_bgs))
            ax.text(0.08, 0.92, text, ha='left', va='top',
                    transform=ax.transAxes, zorder=200)
            if statistic == 'ds':
                ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
            else:
                ax.set_xlabel(r'$\theta \, [\mathrm{arcmin}]$')

            if j > 0:
                plt.setp(ax.get_yticklabels(), visible=False)

        if statistic == 'ds':
            axes[-1][0].set_ylabel(
                r'$r_p \Delta\Sigma \, [10^6 \, M_\odot / \mathrm{pc}]$')
        else:
            axes[-1][0].set_ylabel(
                r'$\theta \gamma_t \, [10^3 \, \mathrm{arcmin}]$')

        axes[i][0].set_xscale('log')

        axes[-1].append(fig.add_subplot(gs[i, -1]))
        colors.append(plt.get_cmap('plasma')(
            np.linspace(0.0, 0.8, len(table_l_1[i][0]))))
        cmap = mpl.colors.ListedColormap(colors[-1])
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm._A = []
        tick_label_list = [['DES', 'HSC', 'KiDS'][i] + '-{}'.format(k + 1) for
                           k in range(len(table_l_1[i][0]))]
        ticks = np.linspace(0, 1, len(tick_label_list) + 1)
        ticks = 0.5 * (ticks[1:] + ticks[:-1])
        cb = plt.colorbar(sm, cax=axes[-1][-1], pad=0.0, ticks=ticks)
        cb.ax.set_yticklabels(tick_label_list)
        cb.ax.minorticks_off()
        cb.ax.tick_params(size=0)

        for j in range(n_bins_l):
            stacking_kwargs = zebu.stacking_kwargs(survey, statistic=statistic)
            if statistic == 'ds':
                function = excess_surface_density
            elif statistic == 'gt':
                function = tangential_shear
            else:
                raise ValueError("Unknown statistic '{}'.".format(statistic))
            for k in range(len(table_l_1[i][0])):

                z_l_max = np.concatenate([
                    zebu.LENS_Z_BINS['bgs'], zebu.LENS_Z_BINS['lrg']])[j + 1]
                z_l_min = zebu.SOURCE_Z_BINS[survey_list[i]][k]
                if z_l_max >= z_l_min:
                    continue

                if kwargs_1 != kwargs_2:
                    y = difference(
                        table_l_1[i][j][k], table_l_2=table_l_2[i][j][k],
                        table_r=table_r_1[i][j][k],
                        table_r_2=table_r_2[i][j][k], function=function,
                        stacking_kwargs=stacking_kwargs)
                    y_cov = jackknife_resampling(
                        difference, table_l_1[i][j][k],
                        table_l_2=table_l_2[i][j][k],
                        table_r=table_r_1[i][j][k],
                        table_r_2=table_r_2[i][j][k], function=function,
                        stacking_kwargs=stacking_kwargs)
                else:
                    y = function(
                        table_l_1[i][j][k], table_r=table_r_1[i][j][k],
                        **stacking_kwargs)
                    y_cov = jackknife_resampling(
                        function, table_l_1[i][j][k],
                        table_r=table_r_1[i][j][k], **stacking_kwargs)

                if statistic == 'gt':
                    y *= 1e3
                    y_cov *= 1e6

                y_err = np.sqrt(np.diag(y_cov))

                plotline, caps, barlinecols = axes[i][j].errorbar(
                    x, x * y, yerr=x * y_err, fmt='-o', ms=2,
                    color=colors[i][k], zorder=k + 100)
                plt.setp(barlinecols[0], capstyle='round')

    fig.suptitle(title, fontsize=16, y=0.99)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0.3)
    plt.savefig(file_stem + '.pdf')
    plt.savefig(file_stem + '.png', dpi=300)

    plt.close()


for statistic in ['ds', 'gt']:
    plot_results('intrinsic_' + statistic, statistic=statistic,
                 config=dict(photometric_redshifts=False),
                 title='Intrinsic Gravitational Signal')
    plot_results('photometric_redshift_' + statistic, statistic=statistic,
                 config=dict(photometric_redshifts=(False, True)),
                 title='Photometric Redshifts')
    plot_results('lens_magnification_' + statistic, statistic=statistic,
                 config=dict(lens_magnification=(False, True)),
                 title='Lens Magnification')
