import os
import zebu
import fitsio
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, hstack
from dsigma.helpers import dsigma_table
from dsigma.physics import critical_surface_density

# %%

fig, ax_list = plt.subplots(figsize=(7, 2), ncols=3, sharex=True,
                            sharey=True)
survey_list = ['DES', 'HSC', 'KiDS']
z_l_list = [0.2, 0.4, 0.6, 0.8]
color_list = plt.get_cmap('plasma')([0.0, 0.25, 0.5, 0.75])


for ax, survey in zip(ax_list, survey_list):

    ax.text(0.5, 0.95, survey, horizontalalignment='center',
            verticalalignment='top', transform=ax.transAxes)

    print(survey)

    if survey == 'DES':

        table_s = []
        fname_list = ['mcal-y1a1-combined-riz-unblind-v4-matched.fits',
                      'y1a1-gold-mof-badregion_BPZ.fits',
                      'mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits']
        columns_list = [['e1', 'e2', 'R11', 'R12', 'R21', 'R22', 'region',
                         'ra', 'dec', 'flags_select', 'flags_select_1p',
                         'flags_select_1m', 'flags_select_2p',
                         'flags_select_2m'],
                        ['Z_MC'], ['MEAN_Z']]

        for fname, columns in zip(fname_list, columns_list):
            table_s.append(Table(fitsio.read(os.path.join(
                zebu.base_dir, 'lwb', 'des', 'raw', fname), columns=columns),
                names=columns))
        table_s = hstack(table_s)
        table_s = table_s[table_s['flags_select'] == 0]

    if survey == 'HSC':
        table_s = Table.read(os.path.join(
            zebu.base_dir, 'lwb', 'hsc', 'raw',
            'hsc_s16a_lensing.fits'))
    elif survey == 'KiDS':
        table_s = Table.read(os.path.join(
            zebu.base_dir, 'lwb', 'kids', 'raw',
            'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits'))

    table_s = dsigma_table(table_s, 'source', survey=survey)
    table_s = table_s[table_s['z'] < zebu.source_z_bins[survey.lower()][-1]]

    for z_l, color in zip(z_l_list, color_list):

        sigma_crit = critical_surface_density(z_l, table_s['z'],
                                              cosmology=zebu.cosmo)

        dz_list = np.linspace(0, 0.9, 91)
        error = np.zeros(len(dz_list))

        for i, dz in enumerate(dz_list):

            use = table_s['z'] > z_l + dz
            w_ls = table_s['w'] / sigma_crit**2
            w_ls_sigma_crit = table_s['w'] / sigma_crit
            e_t = 0.5 * (table_s['e_1'] + table_s['e_2'])

            if np.sum(w_ls[use]) == 0:
                error[i] = np.inf
            else:
                error[i] = (np.sqrt(np.sum(
                    w_ls_sigma_crit[use]**2 * e_t[use]**2)) / np.sum(w_ls[use]))

        ax.plot(dz_list, error[0] / error, color=color,
                label=r'$z_l = {:.1f}$'.format(z_l) if survey == 'HSC' else '')

    ax.set_xlabel(r'$\Delta z_{\rm min}$')
    ax.set_xlim(np.amin(dz_list), np.amax(dz_list))
    ax.set_ylim(ymin=0)

    if survey == 'HSC':
        ax.legend(loc='lower left', frameon=False)

ax_list[0].set_ylabel(r'Relative S/N')
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('redshift_cut.pdf')
plt.savefig('redshift_cut.png', dpi=300)
