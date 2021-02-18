import os
import zebu
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import jackknife_resampling

# %%

z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]
stacking_kwargs = zebu.stacking_kwargs(0)

# %%

for ext in ['', '_gamma', '_gamma_zspec', '_gamma_zspec_w_sys']:

    for lens_bin in range(4):

        fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)

        for source_bin in range(4):

            try:
                fname_l = 'l{}_s{}_l{}.hdf5'.format(lens_bin, source_bin, ext)
                fname_r = 'l{}_s{}_r{}.hdf5'.format(lens_bin, source_bin, ext)
                table_l = Table.read(os.path.join('jackknife', fname_l),
                                     path='data')
                table_r = Table.read(os.path.join('jackknife', fname_r),
                                     path='data')
            except FileNotFoundError:
                continue

            stacking_kwargs['table_r'] = table_r
            stacking_kwargs['return_table'] = True
            delta_sigma = excess_surface_density(table_l, **stacking_kwargs)
            stacking_kwargs['return_table'] = False
            delta_sigma['ds_err'] = np.sqrt(np.diag(
                jackknife_resampling(excess_surface_density, table_l,
                                     **stacking_kwargs)))

            color = 'C{}'.format(source_bin)

            axarr[0].plot(delta_sigma['rp'], delta_sigma['f_bias'],
                          color=color, ls='-', label=r"$f_{\rm bias}$" if
                          source_bin == 3 else "")
            axarr[0].plot(delta_sigma['rp'], delta_sigma['b'], color=color,
                          ls='--', label=r"boost" if source_bin == 3 else "")
            axarr[1].errorbar(
                delta_sigma['rp'] * (1 + (source_bin - lens_bin) * 0.03),
                delta_sigma['rp'] * delta_sigma['ds'], color=color,
                label=r'${:.1f} < z_s < {:.1f}$'.format(
                    z_bins[source_bin + 2], z_bins[source_bin + 3]),
                yerr=delta_sigma['rp'] * delta_sigma['ds_err'],
                fmt='.', ms=0)

            delta_sigma.write('results/result_{}_{}{}.csv'.format(
                lens_bin, source_bin, ext), overwrite=True)

        axarr[0].set_title(r'${:.1f} < z_l < {:.1f}$'.format(
            z_bins[lens_bin], z_bins[lens_bin + 1]))
        axarr[0].legend(loc='upper right', ncol=2)
        axarr[1].legend(loc='upper left', frameon=False)
        axarr[0].set_ylabel(r'Corrections')
        axarr[1].set_ylim(ymin=0)
        axarr[1].set_xlabel(
            r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        axarr[1].set_ylabel(
            r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')

        plt.xscale('log')
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0)
        plt.savefig('results/result_{}{}.pdf'.format(lens_bin, ext))
        plt.savefig('results/result_{}{}.png'.format(lens_bin, ext), dpi=300)
        plt.close()

# %%

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

for lens_bin in range(4):
    for source_bin in range(4):

        try:
            table_l_1 = Table.read(os.path.join(
                'jackknife', 'l{}_s{}_l_gamma.hdf5'.format(
                    lens_bin, source_bin)), path='data')
            table_r_1 = Table.read(os.path.join(
                'jackknife', 'l{}_s{}_r_gamma.hdf5'.format(
                    lens_bin, source_bin)), path='data')
            table_l_2 = Table.read(os.path.join(
                'jackknife', 'l{}_s{}_l_gamma_zspec_w_sys.hdf5'.format(
                    lens_bin, source_bin)), path='data')
            table_r_2 = Table.read(os.path.join(
                'jackknife', 'l{}_s{}_r_gamma_zspec_w_sys.hdf5'.format(
                    lens_bin, source_bin)), path='data')
        except FileNotFoundError:
            continue

        stacking_kwargs['table_r'] = table_r_2
        stacking_kwargs['return_table'] = False
        ds_norm = excess_surface_density(table_l_2, **stacking_kwargs)

        dds = zebu.ds_diff(
            table_l_1, table_r=table_r_1, table_l_2=table_l_2,
            table_r_2=table_r_2, ds_norm=ds_norm)
        dds_cov = jackknife_resampling(
            zebu.ds_diff, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
            table_r_2=table_r_2, ds_norm=ds_norm)
        dds_err = np.sqrt(np.diag(dds_cov))

        i_min = np.arange(len(rp))[rp > 0.5][0]
        dds_ave, dds_ave_cov = zebu.linear_regression(
            rp[i_min:], dds[i_min:], dds_cov[i_min:, i_min:],
            return_err=True)
        dds_ave = dds_ave[0]
        dds_ave_err = np.sqrt(dds_ave_cov[0, 0])
        plt.errorbar(rp[i_min:] * (1 + source_bin * 0.03), dds[i_min:],
                     yerr=dds_err[i_min:],
                     label=r'${:.1f} < z_s < {:.1f}: {:.3f} \pm {:.3f}$'.format(
            zebu.source_z_bins(0)[source_bin],
            zebu.source_z_bins(0)[source_bin + 1],
            dds_ave, dds_ave_err), fmt='.', ms=0)

    plt.title(r'${:.1f} < z_l < {:.1f}$'.format(
        zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))
    plt.legend(loc='upper center', frameon=False)
    plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'$(\Delta \Sigma_{\rm phot} - \Delta \Sigma_{\rm spec}) ' +
               r'/ \Delta \Sigma_{\rm spec}$')

    plt.xscale('log')
    plt.axhline(0.0, ls='--', color='black')
    plt.tight_layout(pad=0.3)
    plt.subplots_adjust(hspace=0)
    fname = 'diff_spec_vs_phot_{}'.format(lens_bin)
    plt.savefig(os.path.join('results', fname + '.pdf'))
    plt.savefig(os.path.join('results', fname + '.png'), dpi=300)
    plt.close()
