import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt

z_bins = [0.15, 0.31, 0.43, 0.54, 0.70]
rp_bins = np.logspace(np.log10(0.05), np.log10(15.), 11)
rp = 0.5 * (rp_bins[1:] + rp_bins[:-1])

# %%

for survey in ['CFHT', 'KiDS']:

    fig, axarr = plt.subplots(figsize=(7, 7), nrows=2, ncols=2, sharex=True,
                              sharey=True)

    for i in range(4):

        ax = axarr[i // 2, i % 2]

        if i > 1:
            ax.set_xlabel(r'$R \, [h^{-1} \, \mathrm{Mpc}]$')
        if i % 2 == 0:
            ax.set_ylabel(r'$R \times \Delta\Sigma \, [\mathrm{Mpc} \, ' +
                          r'M_\odot \, \mathrm{pc}^{-2}]$')

        if survey == 'SDSS' and i > 1:
            continue

        fname = 'DSResultFiles/{}_{}_{}_{}_wtot.dat'.format(
            survey, 'LOWZ' if i < 2 else 'CMASS', z_bins[i], z_bins[i + 1])
        data = np.genfromtxt(fname)
        ax.errorbar(
            rp, rp * data[:, 1], yerr=rp * (
                data[:, 2] if survey != 'CFHT' else data[:, 3]),
            color='magenta', marker='o', ls='', fillstyle='none')

        result = Table.read(os.path.join(
            'results', '{}_{}.csv'.format(survey.lower(), i)))
        ax.errorbar(rp * 1.1, rp * result['ds'], yerr=rp * result['ds_err'],
                    color='royalblue', marker='x', ls='', fillstyle='none')

        ax.annotate(r'{} ${} < z < {}$'.format('LOWZ' if i < 2 else 'CMASS',
                                               z_bins[i], z_bins[i + 1]),
                    xy=(0.95, 0.95), xycoords='axes fraction', ha='right',
                    va='top')
    plt.xscale('log')
    plt.tight_layout(pad=0.8)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(os.path.join('results', '{}_lwb_vs_dsigma.pdf'.format(survey)))
    plt.savefig(os.path.join('results', '{}_lwb_vs_dsigma.png'.format(survey)),
                dpi=300)
    plt.close()
