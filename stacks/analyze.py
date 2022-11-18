import os
import zebu
import numpy as np
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table
from astropy.io.ascii import convert_numpy
from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.stacking import tangential_shear
from dsigma.jackknife import jackknife_resampling

fpath = 'plots'


def read_compute_file(config, lens_bin, source_bin=None, delta_sigma=True):

    fpath = os.path.join('results', '{}'.format(config))

    if delta_sigma:
        fname = 'l{}_ds.hdf5'.format(lens_bin)
        compute = Table.read(os.path.join(fpath, fname))
    else:
        fname = 'l{}_s{}_gt.hdf5'.format(lens_bin, source_bin)
        compute = Table.read(os.path.join(fpath, fname))

    n_pairs = np.sum(compute['sum 1'], axis=1)
    return compute[n_pairs > 0.01 * np.amax(n_pairs)]


def read_compute(lenses, sources, delta_sigma=True, lens_magnification=False,
                 source_magnification=False, fiber_assignment=False,
                 intrinsic_alignment=False, photometric_redshifts=False,
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
    assert np.sum(select) == 1

    config = table['configuration'][select][0]

    compute = []
    for lens_bin in range(len(zebu.LENS_Z_BINS[lenses.split('-')[0]]) - 1):
        if delta_sigma:
            compute.append(read_compute_file(config, lens_bin))
        else:
            compute.append([])
            for source_bin in range(len(zebu.SOURCE_Z_BINS[sources]) - 1):
                compute[-1].append(read_compute_file(
                    config, lens_bin, source_bin, delta_sigma=False))

    return compute


def difference(table_l, table_l_2=None, function=None, stacking_kwargs=None):

    return (function(table_l_2, **stacking_kwargs) -
            function(table_l, **stacking_kwargs))


def plot_difference(separation='physical', statistic='ds', sources=None,
                    relative=True, config={}):

    config_def = dict(
        lens_magnification=False, source_magnification=False,
        fiber_assignment=False, intrinsic_alignment=False,
        photometric_redshifts=False, shear_bias=False, shape_noise=False)

    for key in config_def.keys():
        if key not in config.keys():
            config[key] = config_def[key]

    for key in config.keys():
        if isinstance(config[key], bool):
            config[key] = (config[key], config[key])

    kwargs_1 = {}
    kwargs_2 = {}
    for key in config.keys():
        kwargs_1[key] = config[key][0]
        kwargs_2[key] = config[key][1]

    if separation == 'angle':
        if sources is None:
            raise ValueError('Sources must be specified when plotting as a ' +
                             'function of angle.')
        cat_l_1 = (
            read_compute('bgs', sources, delta_sigma=False, **kwargs_1) +
            read_compute('lrg', sources, delta_sigma=False, **kwargs_1))
        cat_r_1 = (
            read_compute('bgs-r', sources, delta_sigma=False, **kwargs_1) +
            read_compute('lrg-r', sources, delta_sigma=False, **kwargs_1))
        cat_l_2 = (
            read_compute('bgs', sources, delta_sigma=False, **kwargs_2) +
            read_compute('lrg', sources, delta_sigma=False, **kwargs_2))
        cat_r_2 = (
            read_compute('bgs-r', sources, delta_sigma=False, **kwargs_2) +
            read_compute('lrg-r', sources, delta_sigma=False, **kwargs_2))

    elif separation == 'physical':
        cat_l_1 = [[], [], []]
        cat_r_1 = [[], [], []]
        cat_l_2 = [[], [], []]
        cat_r_2 = [[], [], []]
        for i, sources in enumerate(['des', 'hsc', 'kids']):
            for lenses in ['bgs', 'lrg']:
                cat_l_1[i] = cat_l_1[i] + read_compute(
                    lenses, sources, **kwargs_1)
                cat_r_1[i] = cat_l_1[i] + read_compute(
                    lenses + '-r', sources, **kwargs_1)
                cat_l_2[i] = cat_l_2[i] + read_compute(
                    lenses, sources, **kwargs_2)
                cat_r_2[i] = cat_l_2[i] + read_compute(
                    lenses + '-r', sources, **kwargs_2)

    fig = plt.figure(figsize=(7, 2))
    gs = gridspec.GridSpec(1, len(cat_l_1) + 1, wspace=0,
                           width_ratios=[20] * len(cat_l_1) + [1])
    ax_list = []
    for i in range(len(cat_l_1)):
        ax_list.append(fig.add_subplot(
            gs[i], sharex=ax_list[-1] if i > 0 else None,
            sharey=ax_list[-1] if i > 0 else None))
    cax = fig.add_subplot(gs[-1])

    ax_list[0].set_xscale('log')

    lens_list = []
    for lenses in ['bgs', 'lrg']:
        for i in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
            lens_list.append(r'{}-{}'.format(lenses.upper(), i + 1))

    if separation == 'angle':
        text_list = lens_list
        source_list = []
        for i in range(len(zebu.SOURCE_Z_BINS[sources]) - 1):
            source_list.append(r'{}-{}'.format(sources.upper(), i + 1))
    else:
        text_list = ['DES', 'HSC', 'KiDS']

    for ax, text in zip(ax_list, text_list):

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:g}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:+g}\%'.format(y) if y != 0 else r'0\%'))
        ax.axhline(0, ls='--', color='black')

        ax.text(0.08, 0.92, text, ha='left', va='top',
                transform=ax.transAxes, zorder=200)

        if separation == 'angle':
            ax.set_xlabel(r'$\theta \, [\mathrm{arcmin}]$')
        else:
            ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')

    color_list = plt.get_cmap('plasma')(np.linspace(0.0, 0.8, len(cat_l_1[0])))
    cmap = mpl.colors.ListedColormap(color_list)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    if separation == 'angle':
        tick_label_list = source_list
    else:
        tick_label_list = lens_list
    ticks = np.linspace(0, 1, len(tick_label_list) + 1)
    ticks = 0.5 * (ticks[1:] + ticks[:-1])
    cb = plt.colorbar(sm, cax=cax, pad=0.0, ticks=ticks)
    cb.ax.set_yticklabels(tick_label_list)
    cb.ax.minorticks_off()

    if separation == 'angle':
        x = 0.5 * (zebu.THETA_BINS[1:] + zebu.THETA_BINS[:-1])
    else:
        x = 0.5 * (zebu.RP_BINS[1:] + zebu.RP_BINS[:-1])

    for i in range(len(ax_list)):
        for k in range(len(color_list)):

            if separation == 'angle':
                survey = sources
            else:
                survey = ['des', 'hsc', 'kids'][i]

            stacking_kwargs = zebu.stacking_kwargs(survey, statistic=statistic)

            if statistic == 'ds':
                function = excess_surface_density
            elif statistic == 'gt':
                function = tangential_shear
            else:
                raise ValueError("Unknown statistic '{}'.".format(statistic))

            norm = function(cat_l_1[i][k], table_r=cat_r_1[i][k],
                            **stacking_kwargs)
            diff = difference(
                cat_l_1[i][k], table_l_2=cat_l_2[i][k], function=function,
                stacking_kwargs=stacking_kwargs) / norm
            diff_cov = jackknife_resampling(
                difference, cat_l_1[i][k], table_l_2=cat_l_2[i][k],
                function=function, stacking_kwargs=stacking_kwargs) / np.outer(
                    norm, norm)

            if np.all(np.isclose(diff_cov, 0)):
                diff_err = np.zeros(len(norm))
            else:
                diff_err = np.sqrt(np.diag(diff_cov))

            plotline, caps, barlinecols = ax_list[i].errorbar(
                x * (1 + k * 0.05), 100 * diff, yerr=100 * diff_err,
                fmt='o', ms=2, color=color_list[k], zorder=i + 100)
            plt.setp(barlinecols[0], capstyle='round')

    for ax in ax_list[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    return fig, ax_list


fig, ax_list = plot_difference(separation='physical', statistic='ds',
                               config=dict(lens_magnification=(False, True)))
ax_list[0].set_ylabel(r'$\Delta\Sigma$ Lens Magn. Bias')
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig(os.path.join(fpath, 'lens_magnification_ds.pdf'))
plt.savefig(os.path.join(fpath, 'lens_magnification_ds.png'), dpi=300)
plt.close()

for sources in ['des', 'hsc', 'kids']:
    fig, ax_list = plot_difference(
        separation='angle', statistic='gt', sources=sources,
        config=dict(lens_magnification=(False, True)))
    ax_list[0].set_ylabel(r'$\gamma_t$ Lens Magn. Bias')
    ax_list[0].set_ylim(-10, 100)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join(
        fpath, 'lens_magnification_gt_{}.pdf'.format(sources)))
    plt.savefig(os.path.join(
        fpath, 'lens_magnification_gt_{}.png'.format(sources)), dpi=300)
    plt.close()
