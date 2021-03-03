import zebu
import numpy as np
from minisom import MiniSom
from astropy.table import Table
import matplotlib.pyplot as plt

# %%

table_s = Table.read('184787.fits')[::10]

mask = np.ones(len(table_s), dtype=np.bool)
pos = np.zeros((len(table_s), 5))

for i, b in enumerate("grizy"):
    pos[:, i] = table_s['{}cmodel_mag'.format(b)]
    mask = mask & np.isfinite(table_s['{}cmodel_mag'.format(b)])

pos = pos - pos[:, 2][:, np.newaxis]
pos = pos[:, [0, 1, 3, 4]]

pos = pos[mask]
pos = (pos - np.mean(pos, axis=0)) / np.std(pos, axis=0)
table_s = table_s[mask]

# %%

n_x, n_y = 50, 50
som = MiniSom(n_x, n_y, pos.shape[1])
som.pca_weights_init(pos)
som.train(pos, len(pos) * 2, verbose=True)

# %%

plt.imshow(zebu.som_f_of_x(som, pos, table_s['photoz_best'], f=np.median),
           vmax=2.0)
cb = plt.colorbar()
cb.set_label(r'Redshift ${\rm med} (z_{\rm phot})$')
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.3)
plt.savefig('z_median.pdf')
plt.savefig('z_median.png', dpi=300)
plt.close()

# %%

plt.imshow(zebu.som_f_of_x(som, pos, table_s['photoz_best'], f=np.std))
cb = plt.colorbar()
cb.set_label(r'Scatter $\sigma_{z_{\rm phot}}$')
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.3)
plt.savefig('z_scatter.pdf')
plt.savefig('z_scatter.png', dpi=300)
plt.close()

# %%

plt.imshow(zebu.som_f_of_x(som, pos, table_s['icmodel_mag'], f=np.median))
cb = plt.colorbar()
cb.set_label(r'$i$-band Flux ${\rm med}(i_{\rm cmodel})$')
plt.xticks([])
plt.yticks([])
plt.tight_layout(pad=0.3)
plt.savefig('i_median.pdf')
plt.savefig('i_median.png', dpi=300)
plt.close()
