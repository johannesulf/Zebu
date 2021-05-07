import numpy as np
from astropy.table import Table

# %%

z_bins = [0.1, 0.3, 0.5]

for survey in ['hsc', 'kids']:
    for z_min, z_max in zip(z_bins[:-1], z_bins[1:]):

        fname_base = '{:.1f}_{:.1f}_{}'.format(z_min, z_max, survey).replace(
            '.', 'p')

        ds = Table.read(fname_base + '.csv')['ds']
        ds_cov = np.genfromtxt(fname_base + '_cov.csv')
        sn = np.sqrt(np.inner(np.inner(ds, np.linalg.inv(ds_cov)), ds))

        print(np.sqrt(np.sum(ds**2 / np.diag(ds_cov))))
        print(ds / np.sqrt(np.diag(ds_cov)))

        print('{}, {:.1f} < z < {:.1f}: {:.1f}'.format(survey, z_min, z_max, sn))
