import os
import zebu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table

# %%

source_bin = 1
survey = 'kids'

table_s_mock = {}

for survey in ['des', 'hsc', 'kids']:
    table_s_mock[survey] = []
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        table_s_mock[survey].append(zebu.read_mock_data(
            'source', source_bin, survey=survey, magnification=False))

# %%

path = os.path.join(zebu.base_dir, 'mocks')

table_c_data = {}
table_s_data = {}
mag_i_data = {}
z_data = {}

for survey in ['des', 'hsc', 'kids']:

    if survey in ['kids', 'hsc']:
        fname = '{}_cal.fits'.format(survey.lower())
    elif survey == 'des':
        fname = 'des_metacal_cal.fits'

    table_c_data[survey] = Table.read(os.path.join(path, fname))

    if survey == 'des':
        table_c_data[survey].rename_column('zphot', 'z')
        table_c_data[survey].rename_column('zmc', 'z_true')
        table_c_data[survey].rename_column('weinz', 'w_sys')

    elif survey == 'hsc':
        table_c_data[survey].rename_column('redhsc', 'z')
        table_c_data[survey].rename_column('redcosmos', 'z_true')
        table_c_data[survey]['w_sys'] = (table_c_data[survey]['weisom'] *
                                         table_c_data[survey]['weilens'])

    elif survey == 'kids':
        table_c_data[survey].rename_column('z_B', 'z')
        table_c_data[survey].rename_column('z_spec', 'z_true')
        table_c_data[survey].rename_column('spec_weight_CV', 'w_sys')

    table_c_data[survey].keep_columns(['z', 'z_true', 'w_sys'])

    if survey in ['kids', 'hsc']:
        fname = '{}_mag.fits'.format(survey.lower())
    elif survey == 'des':
        fname = 'des_metacal_mag.fits'

    table_s_data[survey] = Table.read(os.path.join(path, fname))

    if survey == 'des':
        table_s_data[survey]['mag'] = 30 - 2.5 * np.log10(
            table_s_data[survey]['flux_i'])
        table_s_data[survey].rename_column('zphotmof', 'z')

    elif survey == 'hsc':
        table_s_data[survey]['mag'] = table_s_data[survey]['icmodel_mag']
        table_s_data[survey].rename_column('photoz_best', 'z')

    elif survey == 'kids':
        table_s_data[survey]['mag'] = table_s_data[survey]['MAG_GAAP_i']
        table_s_data[survey].rename_column('Z_B', 'z')

# %%

fig, axarr = plt.subplots(figsize=(7, 3.5), ncols=3, nrows=2, sharex='row',
                          sharey='row')
z_bins = np.linspace(0, 2, 41)
mag_bins = np.linspace(18, 25, 36)

cmap_data = matplotlib.cm.get_cmap('viridis')
cmap_mock = matplotlib.cm.get_cmap('viridis')
color_data = cmap_data(np.linspace(0.0, 1, 5))
color_mock = cmap_mock(np.linspace(0.0, 1, 5))

for ax, survey in zip(axarr.T, ['des', 'hsc', 'kids']):

    for i in range(2):
        ax[i].text(0.95, 0.95, survey.upper() if survey != 'kids' else 'KiDS',
                   transform=ax[i].transAxes, ha='right', va='top')

    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        ax[0].hist(table_s_mock[survey][source_bin]['z_true'],
                   density=True, color=color_mock[source_bin], bins=z_bins,
                   histtype='step')
        i_band = table_s_mock[survey][source_bin].meta['bands'].index('i')
        ax[1].hist(table_s_mock[survey][source_bin]['mag'][:, i_band],
                   density=True, color=color_mock[source_bin], bins=mag_bins,
                   histtype='step')

        z_min = zebu.source_z_bins[survey][source_bin]
        z_max = zebu.source_z_bins[survey][source_bin + 1]
        use = ((z_min <= table_c_data[survey]['z']) &
               (table_c_data[survey]['z'] < z_max))
        ax[0].hist(table_c_data[survey]['z_true'][use],
                   weights=table_c_data[survey]['w_sys'][use],
                   density=True, color=color_data[source_bin], bins=z_bins,
                   histtype='step', ls='--')

        use = ((z_min <= table_s_data[survey]['z']) &
               (table_s_data[survey]['z'] < z_max))
        ax[1].hist(table_s_data[survey]['mag'][use],
                   density=True, color=color_data[source_bin], bins=mag_bins,
                   histtype='step', ls='--')

        ax[0].set_xlabel(r'Redshift $z$')
        ax[1].set_xlabel(r'Magnitude $m_i$')

axarr[0, 0].set_ylabel('Distribution')
axarr[1, 0].set_ylabel('Distribution')
axarr[0, 0].set_xlim(0, 1.95)
axarr[0, 0].set_ylim(ymin=0)
axarr[1, 0].set_ylim(ymin=0)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('mock_vs_data.pdf')
plt.savefig('mock_vs_data.png', dpi=300)
plt.close()
