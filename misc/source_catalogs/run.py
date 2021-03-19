import os
import zebu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.table import Table, vstack

# %%

source_bin = 1
survey = 'kids'

table_s = {}

for survey in ['des', 'hsc', 'kids']:
    table_s[survey] = []
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        table_s[survey].append(zebu.read_mock_data(
            'source', source_bin, survey=survey, magnification=False))

# %%

table_c = {}

for survey in ['des', 'hsc', 'kids']:
    path = os.path.join(zebu.base_dir, 'mocks')

    if survey in ['kids', 'hsc']:
        fname = '{}_cal.fits'.format(survey.lower())
    elif survey == 'des':
        fname = 'des_metacal_cal.fits'

    table_c[survey] = Table.read(os.path.join(path, fname))

    if survey == 'des':

        table_c[survey].rename_column('zphot', 'z')
        table_c[survey].rename_column('zmc', 'z_true')
        table_c[survey].rename_column('weinz', 'w_sys')

    elif survey == 'hsc':

        table_c[survey].rename_column('redhsc', 'z')
        table_c[survey].rename_column('redcosmos', 'z_true')
        table_c[survey]['w_sys'] = (table_c[survey]['weisom'] *
                                    table_c[survey]['weilens'])

    elif survey == 'kids':

        table_c[survey].rename_column('z_B', 'z')
        table_c[survey].rename_column('z_spec', 'z_true')
        table_c[survey].rename_column('spec_weight_CV', 'w_sys')

    table_c[survey].keep_columns(['z', 'z_true', 'w_sys'])

# %%

fig, axarr = plt.subplots(figsize=(7, 2.0), ncols=3, sharey=True, sharex=True)
z_bins = np.linspace(0, 2, 51)
z_plot = np.linspace(0, 2, 1000)

cmap = matplotlib.cm.get_cmap('viridis')
color = cmap(np.linspace(0, 1, 5))

for ax, survey in zip(axarr, ['des', 'hsc', 'kids']):
    ax.text(0.95, 0.95, survey.upper() if survey != 'kids' else 'KiDS',
            transform=ax.transAxes, ha='right', va='top')
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        n = np.histogram(table_s[survey][source_bin]['z_true'], bins=z_bins)[0]
        n = n / np.sum(n) / np.diff(z_bins)
        n_plot = interp1d(0.5 * (z_bins[1:] + z_bins[:-1]), n, kind='cubic',
                          fill_value=0, bounds_error=False)(z_plot)
        ax.plot(z_plot, n_plot, color=color[source_bin])

        z_min = zebu.source_z_bins[survey][source_bin]
        z_max = zebu.source_z_bins[survey][source_bin + 1]
        use = (z_min <= table_c[survey]['z']) & (table_c[survey]['z'] < z_max)
        ax.hist(table_c[survey]['z_true'][use],
                weights=table_c[survey]['w_sys'][use],
                density=True, color=color[source_bin], bins=z_bins,
                histtype='step')

        ax.set_xlabel(r'Redshift $z$')

axarr[0].set_ylabel('Distribution $n(z)$')
plt.xlim(0, 1.95)
plt.ylim(ymin=0)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('redshift.pdf')
plt.savefig('redshift.png', dpi=300)
plt.close()

# %%

fig, axarr = plt.subplots(figsize=(7, 2.0), ncols=3, sharey=True, sharex=True)
mag_bins = np.linspace(18, 25, 71)
mag_plot = np.linspace(18, 25, 1000)

cmap = matplotlib.cm.get_cmap('viridis')
color = cmap(np.linspace(0, 1, 5))

for ax, survey, band in zip(axarr, ['des', 'hsc', 'kids'], ['r', 'i', 'r']):
    ax.text(0.05, 0.95, survey.upper() if survey != 'kids' else 'KiDS',
            transform=ax.transAxes, ha='left', va='top')
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        i_band = table_s[survey][source_bin].meta['bands'].index(band)

        n = np.histogram(table_s[survey][source_bin]['mag'][:, i_band],
                         bins=mag_bins)[0]
        n = n / np.sum(n) / np.diff(mag_bins)
        n_plot = interp1d(
            0.5 * (mag_bins[1:] + mag_bins[:-1]), n, kind='linear',
            fill_value=0, bounds_error=False)(mag_plot)
        ax.plot(mag_plot, n_plot, color=color[source_bin])

        ax.set_xlabel(r'Magnitude $m_{}$'.format(band))

axarr[0].set_ylabel('Distribution $n(m)$')
plt.xlim(18, 25)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('magnitude.pdf')
plt.savefig('magnitude.png', dpi=300)

# %%

survey = 'gen'
table_s = Table()

for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
    table_s = vstack([table_s, zebu.read_mock_data(
        'source', source_bin, magnification=False)])

fig, axarr = plt.subplots(figsize=(7, 3.0), ncols=2, sharey=True, sharex=True)
mag_bins = np.linspace(18, 26, 41)

for ax, band in zip(axarr, ['r', 'i']):
    i_band = table_s.meta['bands'].index(band)
    ax.hist(table_s['mag'][:, i_band], bins=mag_bins, color='black',
            histtype='step', density=True)
    ax.set_xlabel(r'$m_{}$'.format(band))

axarr[0].set_ylabel('Distribution $n(m)$')
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('magnitude_gen.pdf')
plt.savefig('magnitude_gen.png', dpi=300)
