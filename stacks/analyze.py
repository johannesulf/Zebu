import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import zebu

from astropy.table import Table, vstack
from astropy.io.ascii import convert_numpy
from dsigma.jackknife import jackknife_resampling
from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.stacking import tangential_shear
from matplotlib import gridspec
from pathlib import Path

path = Path('plots')


def read_compute_file(config, lens_bin, source_bin=None, delta_sigma=True):

    path = Path('results', '{}'.format(config))

    if delta_sigma:
        try:
            fname = 'l{}_ds.hdf5'.format(lens_bin)
            compute = Table.read(path / fname)
        except FileNotFoundError:
            source_bin = 0
            fname = 'l{}_s{}_ds.hdf5'
            compute = []
            while (path / fname.format(lens_bin, source_bin)).exists():
                compute.append(Table.read(path / fname.format(
                    lens_bin, source_bin)))
                source_bin += 1
            meta = compute[0].meta
            compute = vstack(compute)
            compute.meta = meta
    else:
        fname = 'l{}_s{}_gt.hdf5'.format(lens_bin, source_bin)
        compute = Table.read(path / fname)

    n_pairs = np.sum(compute['sum 1'], axis=1)
    return compute[n_pairs > 0.01 * np.amax(n_pairs)]


def read_compute(lenses, sources, delta_sigma=True, lens_magnification=False,
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


def difference(table_l, table_l_2=None, table_r=None, table_r_2=None,
               function=None, stacking_kwargs=None):

    return (function(table_l_2, table_r=table_r_2, **stacking_kwargs) -
            function(table_l, table_r=table_r, **stacking_kwargs))


def plot_results(statistic='ds', sources=None, config={}, title=None):

    config = dict(
        lens_magnification=False, source_magnification=False,
        fiber_assignment=False, intrinsic_alignment=False,
        photometric_redshifts=True, shear_bias=False,
        shape_noise=False) | config

    kwargs_1 = {}
    kwargs_2 = {}
    for key in config.keys():
        if isinstance(config[key], tuple):
            kwargs_1[key] = config[key][0]
            kwargs_2[key] = config[key][1]
        else:
            kwargs_1[key] = config[key]
            kwargs_2[key] = config[key]

    if statistic in ['ds', 'ds_lm']:
        table_l_1 = [[], [], []]
        table_r_1 = [[], [], []]
        table_l_2 = [[], [], []]
        table_r_2 = [[], [], []]
        for i, sources in enumerate(['des', 'hsc', 'kids']):
            for lenses in ['bgs', 'lrg']:
                table_l_1[i] = table_l_1[i] + read_compute(
                    lenses, sources, **kwargs_1)
                table_r_1[i] = table_r_1[i] + read_compute(
                    lenses + '-r', sources, **kwargs_1)
                table_l_2[i] = table_l_2[i] + read_compute(
                    lenses, sources, **kwargs_2)
                table_r_2[i] = table_r_2[i] + read_compute(
                    lenses + '-r', sources, **kwargs_2)
    else:
        if sources is None:
            raise ValueError(
                'Sources must be specified for statistic {}.'.format(
                    statistic))
        table_l_1 = (
            read_compute('bgs', sources, delta_sigma=False, **kwargs_1) +
            read_compute('lrg', sources, delta_sigma=False, **kwargs_1))
        table_r_1 = (
            read_compute('bgs-r', sources, delta_sigma=False, **kwargs_1) +
            read_compute('lrg-r', sources, delta_sigma=False, **kwargs_1))
        table_l_2 = (
            read_compute('bgs', sources, delta_sigma=False, **kwargs_2) +
            read_compute('lrg', sources, delta_sigma=False, **kwargs_2))
        table_r_2 = (
            read_compute('bgs-r', sources, delta_sigma=False, **kwargs_2) +
            read_compute('lrg-r', sources, delta_sigma=False, **kwargs_2))

    fig = plt.figure(figsize=(7, 2.3))
    gs = gridspec.GridSpec(1, len(table_l_1) + 1, wspace=0,
                           width_ratios=[20] * len(table_l_1) + [1])
    ax_list = []
    for i in range(len(table_l_1)):
        ax_list.append(fig.add_subplot(
            gs[i], sharex=ax_list[-1] if i > 0 else None,
            sharey=ax_list[-1] if i > 0 else None))
    cax = fig.add_subplot(gs[-1])

    ax_list[0].set_xscale('log')

    lens_list = []
    for lenses in ['bgs', 'lrg']:
        for i in range(len(zebu.LENS_Z_BINS[lenses]) - 1):
            lens_list.append(r'{}-{}'.format(lenses.upper(), i + 1))

    if statistic in ['ds', 'ds_lm']:
        text_list = ['DES', 'HSC', 'KiDS']
    else:
        text_list = lens_list
        source_list = []
        for i in range(len(zebu.SOURCE_Z_BINS[sources]) - 1):
            source_list.append(r'{}-{}'.format(sources.upper(), i + 1))

    for ax, text in zip(ax_list, text_list):
        ax.axhline(0, ls='--', color='black')
        ax.text(0.08, 0.92, text, ha='left', va='top',
                transform=ax.transAxes, zorder=200)
        if statistic in ['ds', 'ds_lm']:
            ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        else:
            ax.set_xlabel(r'$\theta \, [\mathrm{arcmin}]$')

    if statistic in ['ds', 'ds_lm']:
        ax_list[0].set_ylabel(
            r'$r_p \Delta\Sigma \, [10^6 \, M_\odot / \mathrm{pc}]$')
    else:
        ax_list[0].set_ylabel(
            r'$\theta \gamma_t \, [10^3 \, \mathrm{arcmin}]$')

    color_list = plt.get_cmap('plasma')(
        np.linspace(0.0, 0.8, len(table_l_1[0])))
    cmap = mpl.colors.ListedColormap(color_list)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    if statistic in ['ds', 'ds_lm']:
        tick_label_list = lens_list
    else:
        tick_label_list = source_list
    ticks = np.linspace(0, 1, len(tick_label_list) + 1)
    ticks = 0.5 * (ticks[1:] + ticks[:-1])
    cb = plt.colorbar(sm, cax=cax, pad=0.0, ticks=ticks)
    cb.ax.set_yticklabels(tick_label_list)
    cb.ax.minorticks_off()
    cb.ax.tick_params(size=0)

    if statistic in ['ds', 'ds_lm']:
        x = 0.5 * (zebu.RP_BINS[1:] + zebu.RP_BINS[:-1])
    else:
        x = 0.5 * (zebu.THETA_BINS[1:] + zebu.THETA_BINS[:-1])

    for i in range(len(ax_list)):
        for k in range(len(color_list)):

            if statistic in ['ds', 'ds_rel', 'ds_lm']:
                survey = ['des', 'hsc', 'kids'][i]
            else:
                survey = sources

            stacking_kwargs = zebu.stacking_kwargs(survey, statistic=statistic)

            if statistic == 'ds':
                function = excess_surface_density
            elif statistic == 'gt':
                function = tangential_shear
            else:
                raise ValueError("Unknown statistic '{}'.".format(statistic))

            if kwargs_1 != kwargs_2:
                y = difference(
                    table_l_1[i][k], table_l_2=table_l_2[i][k],
                    table_r=table_r_1[i][k], table_r_2=table_r_2[i][k],
                    function=function, stacking_kwargs=stacking_kwargs)
                y_cov = jackknife_resampling(
                    difference, table_l_1[i][k], table_l_2=table_l_2[i][k],
                    table_r=table_r_1[i][k], table_r_2=table_r_2[i][k],
                    function=function, stacking_kwargs=stacking_kwargs)
            else:
                y = function(table_l_1[i][k],  table_r=table_r_1[i][k],
                             **stacking_kwargs)
                y_cov = jackknife_resampling(
                    function, table_l_1[i][k], table_r=table_r_1[i][k],
                    **stacking_kwargs)

            if statistic == 'gt':
                y *= 1e3
                y_cov *= 1e6

            y_err = np.sqrt(np.diag(y_cov))

            plotline, caps, barlinecols = ax_list[i].errorbar(
                x, x * y, yerr=x * y_err, fmt='-o', ms=2, color=color_list[k],
                zorder=k + 100)
            plt.setp(barlinecols[0], capstyle='round')

    for ax in ax_list[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    if title is not None:
        ax_list[len(ax_list) // 2].set_title(title)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0.3)

    return fig, ax_list


fig, ax_list = plot_results(
    statistic='ds', config=dict(photometric_redshifts=False),
    title='Intrinsic Signal')
plt.savefig(path / 'intrinsic_ds.pdf')
plt.savefig(path / 'intrinsic_ds.png', dpi=300)
plt.close()

for sources in ['des', 'hsc', 'kids']:
    fig, ax_list = plot_results(
        statistic='gt', sources=sources, title='Intrinsic Signal')
    plt.savefig(path / 'intrinsic_gt_{}.pdf'.format(sources))
    plt.savefig(path / 'intrinsic_gt_{}.png'.format(sources), dpi=300)
    plt.close()


fig, ax_list = plot_results(
    statistic='ds', config=dict(lens_magnification=(False, True)),
    title='Lens Magnification')
plt.savefig(path / 'lens_magnification_ds.pdf')
plt.savefig(path / 'lens_magnification_ds.png', dpi=300)
plt.close()

for sources in ['des', 'hsc', 'kids']:
    fig, ax_list = plot_results(
        statistic='gt', sources=sources,
        config=dict(lens_magnification=(False, True)),
        title='Lens Magnification')
    plt.savefig(path / 'lens_magnification_gt_{}.pdf'.format(sources))
    plt.savefig(path / 'lens_magnification_gt_{}.png'.format(sources), dpi=300)
    plt.close()


fig, ax_list = plot_results(
    statistic='ds', config=dict(source_magnification=(False, True)),
    title='Source Magnification')
plt.savefig(path / 'source_magnification_ds.pdf')
plt.savefig(path / 'source_magnification_ds.png', dpi=300)
plt.close()


fig, ax_list = plot_results(
    statistic='ds', config=dict(photometric_redshifts=(False, True)))
plt.savefig(path / 'photometric_redshift_ds.pdf')
plt.savefig(path / 'photometric_redshift_ds.png', dpi=300)
plt.close()


fig, ax_list = plot_results(
    statistic='ds', config=dict(intrinsic_alignment=(False, True)),
    title='Intrinsic Alignments')
plt.savefig(path / 'intrinsic_alignment_ds.pdf')
plt.savefig(path / 'intrinsic_alignment_ds.png', dpi=300)
plt.close()

for sources in ['des', 'hsc', 'kids']:
    fig, ax_list = plot_results(
        statistic='gt', sources=sources,
        config=dict(intrinsic_alignment=(False, True)),
        title='Intrinsic Alignments')
    plt.savefig(path / 'intrinsic_alignment_gt_{}.pdf'.format(sources))
    plt.savefig(path / 'intrinsic_alignment_gt_{}.png'.format(sources),
                dpi=300)
    plt.close()


fig, ax_list = plot_results(
    statistic='ds', config=dict(shear_bias=(False, True)),
    title='Residual Shear Bias')
plt.savefig(path / 'shear_bias_ds.pdf')
plt.savefig(path / 'shear_bias_ds.png', dpi=300)
plt.close()

for sources in ['des', 'hsc', 'kids']:
    fig, ax_list = plot_results(
        statistic='gt', sources=sources, config=dict(shear_bias=(False, True)),
        title='Residual Shear Bias')
    plt.savefig(path / 'shear_bias_gt_{}.pdf'.format(sources))
    plt.savefig(path / 'shear_bias_gt_{}.png'.format(sources), dpi=300)
    plt.close()
