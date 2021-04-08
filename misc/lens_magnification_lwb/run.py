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

import camb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

from dsigma.physics import critical_surface_density
from dsigma.physics import lens_magnification_shear_bias

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
plt.close()

# %%

table = Table.read('hod_model_saito2016_mdpl2.txt', format='ascii')
rp = table['col1']
ds_full = table['col2'] * 0.8

z_l = 0.6
z_s = 0.8

cosmo = FlatLambdaCDM(70, 0.3)
theta = rp / cosmo.angular_diameter_distance(z_l).value

pars = camb.CAMBparams()
pars.set_cosmology(H0=67.66, ombh2=0.02242, omch2=0.11933)
pars.InitPower.set_params(ns=0.9665, As=2.105e-9)
pars.set_matter_power(redshifts=np.linspace(z_l, 0, 10), kmax=2000.0,
                      nonlinear=True)

camb_results = camb.get_results(pars)

gamma = np.array([lens_magnification_shear_bias(
    theta_i, 3.0, z_l, z_s, camb_results) for theta_i in theta])
ds_lm = gamma * critical_surface_density(z_l, z_s, cosmology=cosmo,
                                         comoving=False)

plt.plot(rp, ds_lm / ds_full)
plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [\mathrm{Mpc}]$')
plt.ylabel(r'$\Delta\Sigma_{\mathrm{lm}} / \Delta\Sigma{\mathrm{tot}}$')
plt.tight_layout(pad=0.8)
plt.savefig('bias.pdf')
plt.savefig('bias.png', dpi=300)
