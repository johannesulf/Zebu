"""
SELECT photo.ra, photo.dec, spec.z, photo.cmodelmag_i, photo.extinction_i, photo.dered_g, photo.dered_r, photo.dered_i
FROM photoprimary as photo
JOIN specObj as spec ON photo.specobjid = spec.specobjid 
WHERE
  (photo.dered_r-photo.dered_i) < 2 AND
  photo.cmodelmag_i-photo.extinction_i BETWEEN 17.5 AND 19.9 AND
  (photo.dered_r-photo.dered_i) - (photo.dered_g-photo.dered_r)/8. > 0.55 AND
  photo.fiber2mag_i < 21.5 AND
  photo.cmodelmag_i-photo.extinction_i < 19.86 + 1.60*((photo.dered_r-photo.dered_i) - (photo.dered_g-photo.dered_r)/8. - 0.80) AND
  spec.z BETWEEN 0.54 AND 0.7
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

# %%

cmass = Table.read('CMASS.fits')

d_perp = ((cmass['dered_r'] - cmass['dered_i']) -
          (cmass['dered_g'] - cmass['dered_r']) / 8)

# %%

flux = 10**(-0.4 * (cmass['cmodelmag_i'] - cmass['extinction_i'] - np.minimum(
    19.9, 19.86 + 1.6 * (d_perp - 0.8))))

use = cmass['z'] > 0.54

flux_min = np.logspace(0, 1, 100)

plt.plot(flux_min, [np.sum(flux[use] > flux_min[i]) for i in
                    range(len(flux_min))])
plt.plot(flux_min, np.sum(use) * flux_min**-1, ls='--', color='black')
plt.plot(flux_min, np.sum(use) * flux_min**-3, ls=':', color='black')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Flux $s_i / s_{i, \rm cut}$')
plt.ylabel(r'Number $N(>s_i)$')
plt.title('CMASS $0.54 - 0.7$')
plt.tight_layout(pad=0.3)
plt.savefig('cmass_flux_dist.pdf')
plt.savefig('cmass_flux_dist.png', dpi=300)
