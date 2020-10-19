import os
import zebu
import itertools
import numpy as np
from astropy.table import Table, vstack
import matplotlib.pyplot as plt
from dsigma.stacking import excess_surface_density
from dsigma.jackknife import jackknife_resampling

surveys = ['hsc', 'kids', 'des']
z_bins_l = [0.1, 0.3, 0.5, 0.7, 0.9]
rp = 0.5 * (zebu.rp_bins[1:] + zebu.rp_bins[:-1])
gamma_list = [False, True]
zspec_list = [False, False]


def read_precompute(survey, lens_bin, source_bin, zspec=False, gamma=False):

    table_l_all = Table()
    table_r_all = Table()

    if source_bin == 'all':
        source_bin_all = np.arange(
            len(zebu.source_z_bins(1, survey=survey)) - 1)
    else:
        source_bin_all = np.atleast_1d(source_bin)

    for source_bin in source_bin_all:

        try:
            fname_l = 'l{}_s{}_{}_l.hdf5'.format(lens_bin, source_bin,
                                                 survey)
            fname_r = 'l{}_s{}_{}_r.hdf5'.format(lens_bin, source_bin,
                                                 survey)

            if gamma:
                fname_l = fname_l[:-5] + '_gamma' + fname_l[-5:]
                fname_r = fname_r[:-5] + '_gamma' + fname_r[-5:]
            if zspec:
                fname_l = fname_l[:-5] + '_zspec' + fname_l[-5:]
                fname_r = fname_r[:-5] + '_zspec' + fname_r[-5:]

            table_l = Table.read(
                os.path.join('precompute', fname_l), path='data')
            table_r = Table.read(
                os.path.join('precompute', fname_r), path='data')

            # The photo-z calibration factors aren't weighted by the number
            # of sources in each source redshift bin. This can be a problem if
            # we combine results from all source bins. So ensure calibration
            # here.
            if np.sum(table_l['sum w_ls']) > 0:
                norm_c = (np.sum(table_l['sum w_ls']) /
                          np.sum(table_l['calib: sum w_ls w_c']))
                table_l['calib: sum w_ls w_c'] *= norm_c
                table_l['calib: sum w_ls w_c ' +
                        'sigma_crit_p / sigma_crit_t'] *= norm_c
                table_r['calib: sum w_ls w_c'] *= norm_c
                table_r['calib: sum w_ls w_c ' +
                        'sigma_crit_p / sigma_crit_t'] *= norm_c

            table_l_all = vstack([table_l, table_l_all])
            table_r_all = vstack([table_r, table_r_all])
            table_l_all.meta['rp_bins'] = table_l.meta['rp_bins']
            table_r_all.meta['rp_bins'] = table_r.meta['rp_bins']
        except FileNotFoundError:
            continue

    return table_l_all, table_r_all


def ds_diff(table_l, table_r=None, table_l_2=None, table_r_2=None,
            survey_1='hsc', survey_2='kids', ds_norm=None):

    ds_1 = excess_surface_density(
        table_l, table_r=table_r, photo_z_dilution_correction=True,
        boost_correction=True, random_subtraction=True,
        shear_bias_correction=(survey_1 != 'des'),
        shear_responsivity_correction=(survey_1 == 'hsc'),
        metacalibration_response_correction=(survey_1 == 'des'))
    ds_2 = excess_surface_density(
        table_l_2, table_r=table_r_2, photo_z_dilution_correction=True,
        boost_correction=True, random_subtraction=True,
        shear_bias_correction=(survey_2 != 'des'),
        shear_responsivity_correction=(survey_2 == 'hsc'),
        metacalibration_response_correction=(survey_2 == 'des'))

    if ds_norm is not None:
        return (ds_1 - ds_2) / ds_norm

    return ds_1 - ds_2


def linear_regression(x, y, cov, fit_constant=True, return_err=False):

    if not fit_constant:
        X = np.vstack([np.ones_like(x), x]).T
    else:
        X = np.atleast_2d(np.ones_like(x)).T

    pre = np.linalg.inv(cov)
    beta_cov = np.linalg.inv(np.dot(np.dot(X.T, pre), X))
    beta = np.dot(np.dot(np.dot(beta_cov, X.T),  pre), y)

    if return_err:
        return beta, beta_cov

    return beta

# %%


print("Calculating all individual results...")

for gamma, zspec in zip(gamma_list, zspec_list):

    print(('Without' if gamma else 'With') + ' Shape Noise...')
    print(('Without' if zspec else 'With') + " Photo-z's...")

    for survey in surveys:

        output = 'results_{}'.format(survey)
        if not os.path.exists(output):
            os.makedirs(output)

        z_bins_s = zebu.source_z_bins(1, survey=survey)

        for lens_bin in range(len(z_bins_l) - 1):

            fig, axarr = plt.subplots(nrows=2, ncols=1, sharex=True)
            data_plotted = False

            for source_bin in range(len(z_bins_s) - 1):

                table_l, table_r = read_precompute(
                    survey, lens_bin, source_bin, gamma=gamma, zspec=zspec)

                if len(table_l) == 0:
                    print("Warning: {}/Lens:{}/Source:{} not finished.".format(
                        survey.upper(), lens_bin, source_bin))
                    continue
                if np.sum(table_l['sum w_ls']) == 0:
                    continue

                kwargs = {
                    'table_r': table_r,
                    'photo_z_dilution_correction': True,
                    'boost_correction': True, 'random_subtraction': True,
                    'return_table': True,
                    'shear_bias_correction': survey != 'des',
                    'shear_responsivity_correction': survey == 'hsc',
                    'metacalibration_response_correction': survey == 'des'}
                delta_sigma = excess_surface_density(table_l, **kwargs)
                kwargs['return_table'] = False
                delta_sigma['ds_err'] = np.sqrt(np.diag(
                    jackknife_resampling(excess_surface_density, table_l,
                                         **kwargs)))

                color = 'C{}'.format(source_bin)

                axarr[0].plot(delta_sigma['rp'], delta_sigma['f_bias'],
                              color=color, ls='-', label=r"$f_{\rm bias}$" if
                              not data_plotted else "")
                axarr[0].plot(delta_sigma['rp'], delta_sigma['b'], color=color,
                              ls='--', label=r"boost" if data_plotted else "")
                data_plotted = True
                axarr[1].errorbar(
                    delta_sigma['rp'] * (1 + (source_bin - lens_bin) * 0.03),
                    delta_sigma['rp'] * delta_sigma['ds'], color=color,
                    label=r'${:.1f} < z_s < {:.1f}$'.format(
                        z_bins_s[source_bin], z_bins_s[source_bin + 1]),
                    yerr=delta_sigma['rp'] * delta_sigma['ds_err'], fmt='.',
                    ms=0)

                fname = 'result_{}_{}'.format(lens_bin, source_bin)
                if gamma:
                    fname = fname + '_gamma'
                if zspec:
                    fname = fname + '_zspec'
                fname = fname + '.csv'

                delta_sigma.write(os.path.join(output, fname), overwrite=True)

            axarr[0].set_title(r'${:.1f} < z_l < {:.1f}$'.format(
                z_bins_l[lens_bin], z_bins_l[lens_bin + 1]))
            if data_plotted:
                axarr[0].legend(loc='upper right', ncol=2)
                axarr[1].legend(loc='upper center', frameon=False, ncol=2,
                                fontsize=8)
            axarr[0].set_ylabel(r'Corrections')
            axarr[1].set_ylim(ymin=0)
            axarr[1].set_xlabel(
                r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
            axarr[1].set_ylabel(
                r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
            axarr[1].set_ylim(0, 10)

            plt.xscale('log')
            plt.tight_layout(pad=0.3)
            plt.subplots_adjust(hspace=0)
            if data_plotted:
                fname = 'result_{}'.format(lens_bin)
                if gamma:
                    fname = fname + '_gamma'
                if zspec:
                    fname = fname + '_zspec'
                plt.savefig(os.path.join(output, fname + '.pdf'))
                plt.savefig(os.path.join(output, fname + '.png'), dpi=300)
            plt.close()

# %%

print('Comparing results for all sources accross surveys...')

output = 'results_all'
if not os.path.exists(output):
    os.makedirs(output)

for gamma, zspec in zip(gamma_list, zspec_list):

    for lens_bin in range(len(z_bins_l) - 1):

        data_plotted = False

        for i, survey in enumerate(surveys):

            z_bins_s = zebu.source_z_bins(1, survey=survey)

            table_l, table_r = read_precompute(survey, lens_bin, 'all',
                                               gamma=gamma, zspec=zspec)

            if len(table_l) == 0 or np.sum(table_l['sum w_ls']) == 0:
                continue

            kwargs = {
                'table_r': table_r, 'photo_z_dilution_correction': True,
                'boost_correction': True, 'random_subtraction': True,
                'return_table': True,
                'shear_bias_correction': survey != 'des',
                'shear_responsivity_correction': survey == 'hsc',
                'metacalibration_response_correction': survey == 'des'}
            delta_sigma = excess_surface_density(table_l, **kwargs)
            kwargs['return_table'] = False
            delta_sigma['ds_err'] = np.sqrt(np.diag(
                jackknife_resampling(excess_surface_density, table_l,
                                     **kwargs)))

            plt.errorbar(
                delta_sigma['rp'] * (1 + i * 0.03), delta_sigma['rp'] *
                delta_sigma['ds'], label=survey.upper(),
                yerr=delta_sigma['rp'] * delta_sigma['ds_err'], fmt='.', ms=0)
            data_plotted = True

        plt.title(r'${:.1f} < z_l < {:.1f}$'.format(
            z_bins_l[lens_bin], z_bins_l[lens_bin + 1]))
        if data_plotted:
            plt.legend(loc='upper left', frameon=False)
        plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')

        plt.xscale('log')
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0)
        fname = 'result_{}'.format(lens_bin)
        if gamma:
            fname = fname + '_gamma'
        if zspec:
            fname = fname + '_zspec'
        if data_plotted:
            plt.savefig(os.path.join(output, fname + '.pdf'))
            plt.savefig(os.path.join(output, fname + '.png'), dpi=300)
        plt.close()

for gamma, zspec in zip(gamma_list, zspec_list):

    for lens_bin in range(len(z_bins_l) - 1):

        i = 0

        table_l, table_r = read_precompute('hsc', lens_bin, 'all')

        if len(table_l) == 0 or np.sum(table_l['sum w_ls']) == 0:
            continue

        ds_norm = excess_surface_density(
            table_l, table_r=table_r, photo_z_dilution_correction=True,
            boost_correction=True, random_subtraction=True,
            shear_bias_correction=True, shear_responsivity_correction=True)

        for survey_1, survey_2 in itertools.combinations(surveys, 2):

            table_l_1, table_r_1 = read_precompute(survey_1, lens_bin, 'all',
                                                   gamma=gamma, zspec=zspec)
            table_l_2, table_r_2 = read_precompute(survey_2, lens_bin, 'all',
                                                   gamma=gamma, zspec=zspec)

            if len(table_l_1) == 0 or np.sum(table_l_1['sum w_ls']) == 0:
                continue
            if len(table_l_2) == 0 or np.sum(table_l_2['sum w_ls']) == 0:
                continue

            dds = ds_diff(
                table_l_1, table_r=table_r_1, table_l_2=table_l_2,
                table_r_2=table_r_2, survey_1=survey_1, survey_2=survey_2,
                ds_norm=ds_norm)
            dds_cov = jackknife_resampling(
                ds_diff, table_l_1, table_r=table_r_1, table_l_2=table_l_2,
                table_r_2=table_r_2, survey_1=survey_1, survey_2=survey_2,
                ds_norm=ds_norm)
            dds_err = np.sqrt(np.diag(dds_cov))
            i_min = np.arange(len(rp))[rp > 1][0]
            dds_ave, dds_ave_cov = linear_regression(
                rp[i_min:], dds[i_min:], dds_cov[i_min:, i_min:],
                return_err=True)
            dds_ave = dds_ave[0]
            dds_ave_err = np.sqrt(dds_ave_cov[0, 0])
            plt.errorbar(rp * (1 + i * 0.03), dds, yerr=dds_err,
                         label=r'{} vs. {}: ${:.3f} \pm {:.3f}$'.format(
                survey_1.upper(), survey_2.upper(), dds_ave, dds_ave_err),
                fmt='.', ms=0)
            i = i + 1

        plt.title(r'${:.1f} < z_l < {:.1f}$'.format(
            z_bins_l[lens_bin], z_bins_l[lens_bin + 1]))
        if i > 0:
            plt.legend(loc='upper left')
        plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        plt.ylabel(r'$(\Delta \Sigma_1 - \Delta \Sigma_2) /' +
                   r'\Delta \Sigma_{\rm norm}$')

        plt.xscale('log')
        plt.xlim(xmin=1)
        plt.ylim(-1, +2)
        plt.axhline(0.0, ls='--', color='black')
        plt.tight_layout(pad=0.3)
        plt.subplots_adjust(hspace=0)
        fname = 'diff_{}'.format(lens_bin)
        if gamma:
            fname = fname + '_gamma'
        if zspec:
            fname = fname + '_zspec'
        plt.savefig(os.path.join(output, fname + '.pdf'))
        plt.savefig(os.path.join(output, fname + '.png'), dpi=300)
        plt.close()
