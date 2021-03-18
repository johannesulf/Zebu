import zebu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import interp1d

# %%

source_bin = 1
survey = 'kids'

table_s = {}

for survey in ['des', 'hsc', 'kids']:
    table_s[survey] = []
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        table_s[survey].append(zebu.read_mock_data(
            'source', source_bin, survey=survey))

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
mag_bins = np.linspace(15, 25, 51)

cmap = matplotlib.cm.get_cmap('viridis')
color = cmap(np.linspace(0, 1, 5))

for ax, survey, band in zip(axarr, ['des', 'hsc', 'kids'], ['r', 'i', 'r']):
    ax.text(0.05, 0.95, survey.upper() if survey != 'kids' else 'KiDS',
            transform=ax.transAxes, ha='left', va='top')
    for source_bin in range(len(zebu.source_z_bins[survey]) - 1):
        i_band = table_s[survey][source_bin].meta['bands'].index(band)
        ax.hist(table_s[survey][source_bin]['mag'][:, i_band],
                bins=mag_bins, color=color[source_bin], histtype='step',
                density=True)
        ax.set_xlabel(r'Magnitude $m_{}$'.format(band))

plt.yscale('log')
axarr[0].set_ylabel('Distribution $n(m)$')
plt.xlim(15, 25)
plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('magnitude.pdf')
plt.savefig('magnitude.png', dpi=300)
