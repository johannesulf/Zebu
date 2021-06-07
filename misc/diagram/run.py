import os
import zebu
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.colors import LogNorm

matplotlib.rc('font', **{'size': 8})


# %%

plt.figure(figsize=(1, 1))
path = os.path.join(zebu.base_dir, 'mocks')
fname = 'kids_cal.fits'
table_c = Table.read(os.path.join(path, fname))
table_c.rename_column('z_B', 'z')
table_c.rename_column('z_spec', 'z_true')
table_c.rename_column('spec_weight_CV', 'w_sys')

plt.hist2d(table_c['z_true'], table_c['z'], bins=np.linspace(0, 2, 51),
           norm=LogNorm(), cmap='inferno')
plt.xlim(0, 2)
plt.ylim(0, 2)
plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])
plt.xlabel(r'$z$', labelpad=-1)
plt.ylabel(r'$\hat{z}$', labelpad=-1)
plt.tight_layout(pad=0.3)
plt.savefig('diagram_redshift.pdf')
plt.savefig('diagram_redshift.png', dpi=300)
plt.close()

# %%

plt.figure(figsize=(1, 1))
fname = 'kids_mag.fits'
table_s = Table.read(os.path.join(path, fname))
table_s['mag'] = table_s['MAG_GAAP_i']
table_s.rename_column('Z_B', 'z')
color_list = plt.get_cmap('plasma')(np.linspace(0.0, 0.9, 5))

for i in range(5):
    use = ((zebu.source_z_bins['kids'][i] < table_s['z']) &
           (table_s['z'] < zebu.source_z_bins['kids'][i + 1]))
    plt.hist(table_s['mag'][use], np.linspace(19, 25, 61), color=color_list[i],
             histtype='step')
plt.yticks([])
plt.xticks([19, 22, 25])
plt.xlabel(r'$m_i$')
plt.ylabel(r'$n(\hat{z}, m_i)$')
plt.tight_layout(pad=0.3)
plt.savefig('diagram_magnitudes.pdf')
plt.savefig('diagram_magnitudes.png', dpi=300)
plt.close()
