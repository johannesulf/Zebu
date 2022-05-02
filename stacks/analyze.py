import os
import zebu
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table

from dsigma.stacking import excess_surface_density, lens_magnification_bias
from dsigma.jackknife import jackknife_resampling

parser = argparse.ArgumentParser()
parser.add_argument('stage', type=int, help='stage of the analysis')
args = parser.parse_args()

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 3
fiber_assignment = args.stage >= 4
intrinsic_alignment = args.stage >= 5
output = 'stage_{}'.format(args.stage)

# %%

if not os.path.isdir(output):
    os.makedirs(output)

if args.stage == 0:
    survey_list = ['gen']
else:
    survey_list = ['des', 'hsc', 'kids']

color_list = plt.get_cmap('plasma')(
    np.linspace(0.0, 0.8, len(zebu.lens_z_bins) - 1))
rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])


def initialize_plot(ylabel=None):

    fig, axarr = plt.subplots(figsize=(7, 2), nrows=1, ncols=3, sharex=True,
                              sharey='row')
    plt.xscale('log')

    for i, ax in enumerate(axarr):

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:g}'.format(y)))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda y, p: r'{:+g}\%'.format(y) if y != 0 else r'0\%'))
        ax.axhline(0, ls='--', color='black')

        ax.text(0.08, 0.92, ['DES', 'HSC', 'KiDS'][i], ha='left', va='top',
                transform=ax.transAxes, zorder=200)

        ax.set_xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        if i == 0:
            if ylabel is None:
                ylabel = r'Residual bias'
            ax.set_ylabel(ylabel)

    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0, hspace=0)

    cmap = mpl.colors.ListedColormap(color_list)
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm._A = []
    cb = plt.colorbar(sm, ax=axarr[-1], pad=0.0, ticks=np.linspace(
        0, 1, len(zebu.lens_z_bins)))
    cb.ax.set_yticklabels(['{:g}'.format(z) for z in zebu.lens_z_bins])
    cb.ax.minorticks_off()
    cb.set_label(r'Lens redshift $z_l$')

    return fig, axarr


def read_precompute(survey, lens_bin, zspec=False, noisy=False,
                    lens_magnification=False, source_magnification=False,
                    fiber_assignment=False, iip_weights=True, runit=False,
                    intrinsic_alignment=False):

    directory = 'precompute'
    fname = 'l{}_{}'.format(lens_bin, survey)

    if noisy:
        fname += '_noisy'
    if zspec:
        fname += '_zspec'
    if runit:
        fname += '_runit'
    if not lens_magnification:
        if not source_magnification:
            fname += '_nomag'
        else:
            fname += '_nolmag'
    if lens_magnification and not source_magnification:
        raise RuntimeError('Precomputation with lens magnification but no ' +
                           'source magnification has not been performed.')
    if not fiber_assignment:
        fname += '_nofib'
    if fiber_assignment and not iip_weights:
        fname += '_noiip'
    if intrinsic_alignment:
        fname += '_ia'

    fname += '.hdf5'

    return Table.read(os.path.join(directory, fname), path='lens')


def difference(table_l, table_l_2=None, survey_1=None, survey_2=None,
               boost=False):

    for survey in [survey_1, survey_2]:
        if survey not in ['gen', 'hsc', 'kids', 'des']:
            raise RuntimeError('Unkown survey!')

    if not boost:
        ds_1 = excess_surface_density(
            table_l, **zebu.stacking_kwargs(survey_1))
        ds_2 = excess_surface_density(
            table_l_2, **zebu.stacking_kwargs(survey_2))
        return ds_1 - ds_2
    else:
        w_1 = np.sum(table_l['sum w_ls'].data *
                     table_l['w_sys'].data[:, None], axis=0)
        w_2 = np.sum(table_l_2['sum w_ls'].data *
                     table_l_2['w_sys'].data[:, None], axis=0)
        return (w_1 * w_2[-1] / w_1[-1] - w_2) / w_1


def plot_difference(ax, color, table_l_1, table_l_2, survey, survey_2=None,
                    ds_norm=1.0, label=None, offset=0, lens_bin=0,
                    boost=False):

    if survey_2 is None:
        survey_2 = survey

    dds = difference(table_l_1, table_l_2=table_l_2, survey_1=survey,
                     survey_2=survey_2, boost=boost) / ds_norm
    dds_cov = jackknife_resampling(
        difference, table_l_1, table_l_2=table_l_2, survey_1=survey,
        survey_2=survey_2, boost=boost)
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
ds_ref = []

for lens_bin in range(len(zebu.lens_z_bins) - 1):

    table_l = read_precompute(
        'gen', lens_bin, zspec=True, lens_magnification=False,
        source_magnification=False, fiber_assignment=False)
    table_l_ref.append(table_l)
    ds_ref.append(excess_surface_density(
        table_l, **zebu.stacking_kwargs('gen')))

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

if args.stage > 0:

    fig, axarr = initialize_plot()

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l, table_l_ref[lens_bin],
                            survey, survey_2='gen', ds_norm=ds_ref[lens_bin],
                            offset=lens_bin)

    ymin, ymax = axarr[0].get_ylim()
    yabsmax = max(abs(ymin), abs(ymax))
    plt.ylim(-yabsmax, +yabsmax)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('reference_comparison')
    plt.close()

if args.stage == 1:

    fig, axarr = initialize_plot()

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_phot = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_spec = read_precompute(
                survey, lens_bin, zspec=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_phot, table_l_spec, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Photometric redshift dilution')
    plt.ylim(-10, +10)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('redshift_dilution_residual')
    plt.close()

    fig, axarr = initialize_plot('Bias')

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_phot = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_spec = read_precompute(
                survey, lens_bin, zspec=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_phot['sum w_ls e_t sigma_crit f_bias'] = \
                table_l_phot['sum w_ls e_t sigma_crit']

            plot_difference(ax, color, table_l_phot, table_l_spec, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Photometric redshift dilution')
    plt.ylim(-60, +60)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('redshift_dilution')
    plt.close()

    fig, axarr = initialize_plot()

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=True, runit=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_runit = read_precompute(
                survey, lens_bin, zspec=True, runit=True,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_l_runit, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Shear response')
    plt.ylim(-10, +10)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('response_residual')
    plt.close()

# %%

if args.stage == 2:

    fig, axarr = initialize_plot('Bias')

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=False,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_l_nomag, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Source magnification')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('source_magnification')
    plt.close()

    fig, axarr = initialize_plot('Bias')

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=False,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_normal, table_l_nomag, survey,
                            offset=lens_bin, boost=True)

    axarr[1].set_title('Impact of magnification on boost estimator')
    plt.ylim(-4, +4)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('boost_magnification_bias')
    plt.close()

# %%

if args.stage == 3:

    camb_results = zebu.get_camb_results()

    fig, axarr = initialize_plot()

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)
            table_l_nomag = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=False,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            ds_lm = lens_magnification_bias(
                table_l_normal, zebu.alpha_l[lens_bin], camb_results,
                photo_z_correction=True)
            ax.plot(rp * (1 + lens_bin * 0.05),
                    100 * ds_lm / ds_ref[lens_bin], color=color, ls='--')

            plot_difference(ax, color, table_l_normal, table_l_nomag, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Lens magnification')
    plt.ylim(-2, +25)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('lens_magnification')
    plt.close()

# %%

if args.stage == 4:

    fig, axarr = initialize_plot('Bias')

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=True, iip_weights=False)
            table_l_nofib = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=False)

            plot_difference(ax, color, table_l_normal, table_l_nofib, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Fibre collisions')
    plt.ylim(-50, +50)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('fiber_collisions')
    plt.close()

    fig, axarr = initialize_plot()

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_normal = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=True)
            table_l_nofib = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=False)

            plot_difference(ax, color, table_l_normal, table_l_nofib, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Fibre collisions')
    plt.ylim(-2, +2)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('fiber_collisions_residual')
    plt.close()

# %%

if args.stage == 5:

    fig, axarr = initialize_plot('Bias')

    for survey, ax in zip(survey_list, axarr):
        for lens_bin, color in enumerate(color_list):

            table_l_ia = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment, intrinsic_alignment=True)
            table_l_noia = read_precompute(
                survey, lens_bin, zspec=False,
                lens_magnification=lens_magnification,
                source_magnification=source_magnification,
                fiber_assignment=fiber_assignment)

            plot_difference(ax, color, table_l_ia, table_l_noia, survey,
                            ds_norm=ds_ref[lens_bin], offset=lens_bin)

    axarr[1].set_title('Intrinsic alignment')
    plt.ylim(-1, +1)
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(wspace=0)
    savefigs('intrinsic_alignment')
    plt.close()
