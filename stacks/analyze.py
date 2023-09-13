import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import zebu

from astropy import units as u
from astropy.table import Table
from astropy.io.ascii import convert_numpy
from dsigma.jackknife import jackknife_resampling
from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.stacking import tangential_shear
from dsigma.stacking import boost_factor as boost_factor_dsigma
from matplotlib import gridspec
from pathlib import Path


zebu.SOURCE_Z_BINS['des'] = np.array([0.0, 0.358, 0.631, 0.872, 2.0])
camb_results = zebu.get_camb_results()
errors = zebu.errors()
for statistic in ['ds', 'gt']:
    for sources in ['des', 'hsc', 'kids']:
        errors[statistic][sources] = np.hstack([
            errors[statistic][sources]['bgs'],
            errors[statistic][sources]['lrg']])
z_l_all = np.concatenate([
    0.5 * (zebu.LENS_Z_BINS['bgs'][1:] + zebu.LENS_Z_BINS['bgs'][:-1]),
    0.5 * (zebu.LENS_Z_BINS['lrg'][1:] + zebu.LENS_Z_BINS['lrg'][:-1])])


def boost_factor(table_l, table_r, **kwargs):
    return 100 * (boost_factor_dsigma(table_l, table_r) - 1)


def read_precomputed_data(
        lenses, sources, statistic, lens_magnification=False,
        source_magnification=False, fiber_assignment=False, iip_weights=True,
        intrinsic_alignment=False, photometric_redshifts=True,
        shear_bias=False, shape_noise=False, reduced_shear=True):

    if sources != 'hsc':
        photometric_redshifts = True

    if lenses in ['bgs-r', 'lrg-r']:
        lens_magnification = False
        fiber_assignment = False

    converters = {'*': [convert_numpy(typ) for typ in (int, float, bool, str)]}
    table = Table.read('config.csv', converters=converters)
    select = np.ones(len(table), dtype=bool)
    select &= table['lenses'] == lenses
    select &= table['sources'] == sources
    select &= table['lens magnification'] == lens_magnification
    select &= table['source magnification'] == source_magnification
    select &= table['fiber assignment'] == fiber_assignment
    if fiber_assignment:
        select &= table['iip weights'] == iip_weights
    select &= table['intrinsic alignment'] == intrinsic_alignment
    select &= table['photometric redshifts'] == photometric_redshifts
    select &= table['shear bias'] == shear_bias
    select &= table['reduced shear'] == reduced_shear
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
            try:
                table = Table.read(path / fname)
                n_pairs = np.sum(table['sum 1'], axis=1)
                table = table[n_pairs > 0.01 * np.amax(n_pairs)]
                data[-1].append(table)
            except FileNotFoundError:
                data[-1].append(None)

    return data


def difference(table_l, table_l_2=None, table_r=None, table_r_2=None,
               function=None, stacking_kwargs=None):

    return (function(table_l_2, table_r=table_r_2, **stacking_kwargs) -
            function(table_l, table_r=table_r, **stacking_kwargs))


def plot_results(path, statistic='ds', survey='des', config={},
                 plot_lens_magnification=False, relative=False,
                 boost_correction=False):

    config = dict(
        lens_magnification=False, source_magnification=False,
        fiber_assignment=False, iip_weights=True, intrinsic_alignment=False,
        photometric_redshifts=True, shear_bias=False,
        shape_noise=False) | config
    config['statistic'] = statistic.split('-')[0]

    kwargs_1 = {}
    kwargs_2 = {}
    for key in config.keys():
        if isinstance(config[key], tuple):
            kwargs_1[key] = config[key][0]
            kwargs_2[key] = config[key][1]
        else:
            kwargs_1[key] = config[key]
            kwargs_2[key] = config[key]

    table_l_1 = (read_precomputed_data('bgs', survey, **kwargs_1) +
                 read_precomputed_data('lrg', survey, **kwargs_1))
    table_r_1 = (read_precomputed_data('bgs-r', survey, **kwargs_1) +
                 read_precomputed_data('lrg-r', survey, **kwargs_1))
    table_l_2 = (read_precomputed_data('bgs', survey, **kwargs_2) +
                 read_precomputed_data('lrg', survey, **kwargs_2))
    table_r_2 = (read_precomputed_data('bgs-r', survey, **kwargs_2) +
                 read_precomputed_data('lrg-r', survey, **kwargs_2))

    n_bins_l = len(table_l_1)
    n_bins_s = len(table_l_1[0])

    fig = plt.figure(figsize=(7, 1.5))
    gs = gridspec.GridSpec(1, n_bins_l + 1, wspace=0,
                           width_ratios=[20] * n_bins_l + [1])
    axes = []

    if statistic.split('-')[0] == 'ds':
        x = 0.5 * (zebu.RP_BINS[1:] + zebu.RP_BINS[:-1])
    else:
        x = 0.5 * (zebu.THETA_BINS[1:] + zebu.THETA_BINS[:-1])

    for j in range(n_bins_l):
        axes.append(fig.add_subplot(
            gs[0, j], sharex=axes[-1] if j > 0 else None,
            sharey=axes[-1] if j > 0 else None))
        ax = axes[-1]
        ax.axhline(0, ls='--', color='black')
        n_bgs = len(zebu.LENS_Z_BINS['bgs']) - 1
        text = '{}-{}'.format('BGS' if j < n_bgs else 'LRG', j + 1 -
                              n_bgs * (j >= n_bgs))
        ax.text(0.08, 0.92, text, ha='left', va='top',
                transform=ax.transAxes, zorder=200)
        if statistic.split('-')[0] == 'ds':
            ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        else:
            ax.set_xlabel(r'$\theta \, [\mathrm{arcmin}]$')

        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:g}'.format(y)))
        if relative or statistic[-5:] == 'boost':
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda y, p: r'{:+g}\%'.format(y) if y != 0 else r'0\%'))
        if relative or statistic[-5:] == 'boost':
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(integer=True, nbins=3))

        if j > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

    if statistic == 'ds':
        if relative:
            axes[0].set_ylabel(r'$\delta \Delta\Sigma$')
        else:
            axes[0].set_ylabel(
                r'$r_p \Delta\Sigma \, [10^6 \, M_\odot / \mathrm{pc}]$')
    elif statistic == 'gt':
        if relative:
            axes[0].set_ylabel(r'$\delta \gamma_t$')
        else:
            axes[0].set_ylabel(
                r'$\theta \gamma_t \, [10^3 \, \mathrm{arcmin}]$')
    else:
        axes[0].set_ylabel(r'Boost Factor $b - 1$')

    axes.append(fig.add_subplot(gs[0, -1]))
    colors = plt.get_cmap('plasma')(np.linspace(0.0, 0.8, n_bins_s))
    cmap = mpl.colors.ListedColormap(colors)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    tick_labels = [(survey.upper() if survey != 'kids' else 'KiDS') +
                   '-{}'.format(k + 1) for k in range(n_bins_s)]
    ticks = np.linspace(0, 1, len(tick_labels) + 1)
    ticks = 0.5 * (ticks[1:] + ticks[:-1])
    cb = plt.colorbar(sm, cax=axes[-1], pad=0.0, ticks=ticks)
    cb.ax.set_yticklabels(tick_labels)
    cb.ax.minorticks_off()
    cb.ax.tick_params(size=0)
    y_all = np.zeros((n_bins_l, n_bins_s, len(x)))

    for j in range(n_bins_l):
        stacking_kwargs = zebu.stacking_kwargs(
            survey, statistic=statistic.split('-')[0])
        stacking_kwargs['boost_correction'] = boost_correction
        if statistic == 'ds':
            function = excess_surface_density
        elif statistic == 'gt':
            function = tangential_shear
        elif statistic in ['ds-boost', 'gt-boost']:
            function = boost_factor
        else:
            raise ValueError("Unknown statistic '{}'.".format(statistic))
        for k in range(n_bins_s):

            z_l = z_l_all[j]
            z_s = np.mean(zebu.SOURCE_Z_BINS[survey][[k, k + 1]])
            if table_l_1[j][k] is None or table_l_2[j][k] is None:
                continue

            if kwargs_1 != kwargs_2:
                y = difference(
                    table_l_1[j][k], table_l_2=table_l_2[j][k],
                    table_r=table_r_1[j][k], table_r_2=table_r_2[j][k],
                    function=function, stacking_kwargs=stacking_kwargs)
                y_cov = jackknife_resampling(
                    difference, table_l_1[j][k], table_l_2=table_l_2[j][k],
                    table_r=table_r_1[j][k], table_r_2=table_r_2[j][k],
                    function=function, stacking_kwargs=stacking_kwargs)
            else:
                y = function(
                    table_l_1[j][k], table_r=table_r_1[j][k],
                    **stacking_kwargs)
                y_cov = jackknife_resampling(
                    function, table_l_1[j][k], table_r=table_r_1[j][k],
                    **stacking_kwargs)

            y_err = np.sqrt(np.diag(y_cov))

            if relative:
                y_norm = function(
                    table_l_1[j][k], table_r=table_r_1[j][k],
                    **stacking_kwargs) / 100
                y /= y_norm
                y_all[j, k, :] = y
                y_err /= y_norm
                y_err = np.abs(y_err)
            else:
                y_all[j, k, :] = y
                if statistic == 'gt':
                    y *= 1e3
                    y_err *= 1e3
                if statistic in ['ds', 'gt']:
                    y = x * y
                    y_err = x * y_err

            if z_l >= z_s - 0.1:
                continue

            plotline, caps, barlinecols = axes[j].errorbar(
                x, y, yerr=y_err, fmt='-o', ms=2, color=colors[k],
                zorder=k + 100)
            plt.setp(barlinecols[0], capstyle='round')

            if plot_lens_magnification:
                y = lens_magnification_bias(
                    table_l_1[j][k], zebu.ALPHA_L[j], camb_results,
                    shear=(statistic.split('-')[0] == 'gt'))
                if relative:
                    y /= y_norm
                else:
                    if statistic == 'gt':
                        y *= 1e3
                    if statistic in ['ds', 'gt']:
                        y = x * y
                axes[j].plot(x, y, ls='--', color=colors[k])

    ymin, ymax = axes[0].get_ylim()
    ymin = min(ymin, -0.01)
    ymax = max(ymax, +0.01)
    ymax = max(ymax, -ymin / 3)
    axes[0].set_ylim(ymin, ymax)
    axes[0].set_xlim(axes[0].get_xlim())

    for j in range(n_bins_l):
        z_l = z_l_all[j]

        if statistic.split('-')[0] == 'ds':
            x_res = 1
        else:
            x_res = (1 * u.Mpc / zebu.COSMOLOGY.comoving_distance(z_l) *
                     u.rad).to(u.arcmin).value
        for ax in axes:
            axes[j].axvspan(0, x_res, color='lightgrey', zorder=-99)

        for k in range(n_bins_s):
            if kwargs_1 != kwargs_2 and statistic in ['ds', 'gt']:
                err = np.copy(errors[statistic][sources][k, j])
                if relative:
                    err = err / y_norm
                else:
                    if statistic == 'gt':
                        err *= 1e3
                    err = x * err
                # axes[j].plot(x, +0.1 * err, color=colors[k], zorder=k + 100,
                #             ls='-', lw=1)
                # axes[j].plot(x, -0.1 * err, color=colors[k], zorder=k + 100,
                #             ls='-', lw=1)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0.3)
    np.save(path.with_suffix('.npy'), y_all, allow_pickle=False)
    plt.savefig(path.with_suffix('.pdf'))
    plt.savefig(path.with_suffix('.png'), dpi=300)

    plt.close()


for path, relative in zip([Path('plots_absolute'), Path('plots_relative')],
                          [False, True]):

    for statistic in ['gt', 'ds']:
        for survey in ['des', 'hsc', 'kids']:

            if not relative:
                plot_results(
                    path / ('gravitational_{}_{}'.format(statistic, survey)),
                    statistic=statistic, survey=survey,
                    config=dict(photometric_redshifts=False),
                    boost_correction=True)

            if survey == 'hsc':
                plot_results(
                    path / ('photometric_redshift_{}_{}'.format(
                        statistic, survey)),
                    statistic=statistic, survey=survey,
                    config=dict(photometric_redshifts=(False, True)),
                    relative=relative)

            if relative:
                plot_results(
                    path / ('boost_{}_{}'.format(statistic, survey)),
                    statistic=statistic + '-boost', survey=survey)

            plot_results(
                path / ('lens_magnification_{}_{}'.format(statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(lens_magnification=(False, True)),
                plot_lens_magnification=True, relative=relative)

            plot_results(
                path / ('source_magnification_{}_{}'.format(
                    statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(source_magnification=(False, True)),
                relative=relative)

            if relative:
                plot_results(
                    path / ('boost_source_{}_{}'.format(statistic, survey)),
                    statistic=statistic + '-boost', survey=survey,
                    config=dict(source_magnification=(False, True)))

            plot_results(
                path / ('intrinsic_alignment_{}_{}'.format(statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(intrinsic_alignment=(False, True)),
                relative=relative)

            plot_results(
                path / ('shear_bias_{}_{}'.format(statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(shear_bias=(False, True)), relative=relative)

            plot_results(
                path / ('fiber_assignment_no_iip_{}_{}'.format(
                    statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(fiber_assignment=(False, True), iip_weights=False),
                relative=relative)

            plot_results(
                path / ('fiber_assignment_{}_{}'.format(statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(fiber_assignment=(False, True)), relative=relative)

            plot_results(
                path / ('reduced_shear_{}_{}'.format(statistic, survey)),
                statistic=statistic, survey=survey,
                config=dict(reduced_shear=(False, True)), relative=relative)
