import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import zebu

from astropy.table import Table
from astropy import units as u

zebu.SOURCE_Z_BINS['des'] = np.array([0.0, 0.358, 0.631, 0.872, 2.0])

# %%


def significance(bias, cov, use=None):

    if use is None:
        use = np.ones(len(bias), dtype=bool)

    bias = bias[use]
    cov = cov[np.outer(use, use)].reshape(np.sum(use), np.sum(use))
    err = np.sqrt(np.diag(cov))

    bias = bias / err
    cov = cov / np.outer(err, err)

    pre = np.linalg.inv(cov)

    return np.dot(np.dot(bias, pre), bias)


# %%

fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(7, 3.5), sharex='row',
                        sharey='row',
                        gridspec_kw={'height_ratios': [0.1, 1, 1]})

z_l = np.array([0.15, 0.25, 0.35, 0.5, 0.7, 0.95])

for i, statistic in enumerate(['gt', 'ds']):
    for j, survey in enumerate(['des', 'hsc', 'kids']):
        ax = axs[i + 1][j]
        ax.axhline(0.1, color='black', ls='--')
        ax.axvspan(0, 1.0, color='lightgrey', zorder=-99)
        ax.set_xlabel(r'Minimum Scale $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
        for name, label in zip([
                'fiber_assignment_no_iip', 'lens_magnification',
                'intrinsic_alignment', 'boost', 'reduced_shear'],
                ['fibre assignment', 'lens magnification',
                 'intrinsic alignment', 'boost factor', 'reduced shear']):

            if name != 'boost':
                data = Table.read(
                    zebu.BASE_PATH / 'stacks' / 'plots_absolute' /
                    '{}_{}_{}.csv'.format(
                        name, statistic, survey))
            else:
                data = Table.read(
                    zebu.BASE_PATH / 'stacks' / 'plots_relative' /
                    '{}_{}_{}.csv'.format(
                        name, statistic, survey))
                data['value'] /= 100
                data['value'] *= Table.read(
                    zebu.BASE_PATH / 'stacks' / 'plots_absolute' /
                    '{}_{}_{}.csv'.format(
                        'gravitational', statistic, survey))['value']

            cov = zebu.covariance(statistic, survey)[0]
            bias = data['value']

            if statistic == 'gt':
                data['r'] = zebu.COSMOLOGY.comoving_distance(z_l).to(
                    u.Mpc).value[data['lens_bin']] * (data['r'] * u.deg).to(
                        u.rad).value

            rp_min = zebu.RP_BINS[:-2]

            if statistic == 'gt':
                rp_min = rp_min[rp_min > 1]

            chi_sq = []
            for k in range(len(rp_min)):
                use = (data['lens_bin'] <= 4) & (data['r'] > rp_min[k])
                if statistic == 'ds':
                    z_s = 0.5 * (
                        zebu.SOURCE_Z_BINS[survey][1:] +
                        zebu.SOURCE_Z_BINS[survey][:-1])
                    use = use & (z_l[data['lens_bin']] <
                                 z_s[data['source_bin']] - 0.4)

                chi_sq.append(significance(bias, cov, use))
            ax.plot(rp_min, chi_sq, label=label)

            text = '{}: '.format(survey.upper() if survey != 'kids' else
                                 'KiDS')

            if statistic == 'gt':
                text += r'$\gamma_t$, no redshift cuts'
            else:
                text += r'$\Delta\Sigma$, redshift cuts'

            ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right',
                    va='top')

for ax in [axs[1][0], axs[2][0]]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin=5e-2)
    ax.set_ylabel(r'Significance $\chi^2$')

for i in range(1, 3):
    plt.setp(axs[1][i].get_yticklabels(), visible=False)
    plt.setp(axs[2][i].get_yticklabels(), visible=False)

handles, labels = axs[1][1].get_legend_handles_labels()
axs[0, 1].legend(handles, labels, loc='center', frameon=False, ncols=3,
                 borderpad=-10, bbox_to_anchor=(0.5, -0.9))
axs[0, 0].axis('off')
axs[0, 1].axis('off')
axs[0, 2].axis('off')

plt.tight_layout(pad=0.3)
plt.savefig('significance.pdf')
plt.savefig('significance.png', dpi=300)
