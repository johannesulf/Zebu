import os
import argparse
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import jackknife_resampling

parser = argparse.ArgumentParser()
parser.add_argument('--survey', help='the lens survey', required=True)
args = parser.parse_args()

# %%

output = 'results_{}'.format(args.survey)
if not os.path.exists(output):
    os.makedirs(output)

z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]

for lens_bin in range(4):

    fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)

    for source_bin in range(4):

        try:
            fname_l = 'l{}_s{}_{}_l.hdf5'.format(lens_bin, source_bin,
                                                 args.survey)
            fname_r = 'l{}_s{}_{}_r.hdf5'.format(lens_bin, source_bin,
                                                 args.survey)
            table_l = Table.read(os.path.join('jackknife', fname_l),
                                 path='data')
            table_r = Table.read(os.path.join('jackknife', fname_r),
                                 path='data')
        except FileNotFoundError:
            continue

        if np.sum(table_l['sum w_ls']) == 0:
            continue

        kwargs = {'table_r': table_r, 'photo_z_dilution_correction': True,
                  'boost_correction': True, 'random_subtraction': True,
                  'return_table': True}
        delta_sigma = excess_surface_density(table_l, **kwargs)
        kwargs['return_table'] = False
        delta_sigma['delta sigma_err'] = np.sqrt(np.diag(
            jackknife_resampling(excess_surface_density, table_l, **kwargs)))

        color = 'C{}'.format(source_bin)

        axarr[0].plot(delta_sigma['rp'], delta_sigma['f_bias'],
                      color=color, ls='-', label=r"$f_{\rm bias}$" if
                      source_bin == 3 else "")
        axarr[0].plot(delta_sigma['rp'], delta_sigma['b'], color=color,
                      ls='--', label=r"boost" if source_bin == 3 else "")
        axarr[1].errorbar(
            delta_sigma['rp'] * (1 + (source_bin - lens_bin) * 0.03),
            delta_sigma['rp'] * delta_sigma['delta sigma'], color=color,
            label=r'${:.1f} < z_s < {:.1f}$'.format(
                z_bins[source_bin + 2], z_bins[source_bin + 3]),
            yerr=delta_sigma['rp'] * delta_sigma['delta sigma_err'], fmt='.',
            ms=0)

        delta_sigma.write(os.path.join(output, 'result_{}_{}.csv'.format(
                                       lens_bin, source_bin)), overwrite=True)

    axarr[0].set_title(r'${:.1f} < z_l < {:.1f}$'.format(
        z_bins[lens_bin], z_bins[lens_bin + 1]))
    axarr[0].legend(loc='upper right', ncol=2)
    axarr[1].legend(loc='upper left', frameon=False)
    axarr[0].set_ylabel(r'Corrections')
    axarr[1].set_ylim(ymin=0)
    axarr[1].set_xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    axarr[1].set_ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')

    plt.xscale('log')
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output, 'result_{}.pdf'.format(lens_bin)))
    plt.savefig(os.path.join(output, 'result_{}.pdf'.format(lens_bin)),
                dpi=300)
    plt.close()
