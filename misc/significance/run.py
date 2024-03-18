import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zebu

from astropy.table import Table
from astropy import units as u
from cycler import cycler

plt.rc('axes', prop_cycle=(
       cycler(color=matplotlib.colormaps['plasma'](np.linspace(0.0, 1.0, 5))) +
       cycler(linestyle=['-', '--', ':', '-.', (0, (5, 1))])))

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

z_l = np.concatenate((zebu.LENS_Z['bgs'], zebu.LENS_Z['lrg'], [0.95]))

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
                    z_s = zebu.SOURCE_Z[survey]
                    use = use & (z_l[data['lens_bin']] <
                                 z_s[data['source_bin']] - 0.2)

                chi_sq.append(significance(bias, cov, use))
            ax.plot(rp_min, chi_sq, label=label)
            table = Table()
            table['rp_min'] = rp_min
            table['delta chi_sq'] = chi_sq

            text = '{}: '.format(survey.upper() if survey != 'kids' else
                                 'KiDS')

            if statistic == 'gt':
                text += r'$\gamma_t$, no redshift cuts'
            else:
                text += r'$\Delta\Sigma$, redshift cuts'

            ax.text(0.95, 0.95, text, transform=ax.transAxes, ha='right',
                    va='top')
            table.write('significance_{}_{}_{}.csv'.format(
                statistic, survey, name), overwrite=True)

for ax in [axs[1][0], axs[2][0]]:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(ymin=5e-2)
    ax.set_ylabel(r'Significance $\Delta\chi^2$')
    ax.set_yticks([0.1, 1, 10, 100], ['0.1', '1', '10', '100'])

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
