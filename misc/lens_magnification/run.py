import camb
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import Planck15
from dsigma.physics import lens_magnification_bias, critical_surface_density

cosmo = Planck15

# %%

tau = 0.0543
As = np.exp(3.0448)*1e-10
ns = 0.96605
H0 = 67.32
h = H0 / 100
wb = 0.022383
wc = 0.12011
mnu = 0.06
kp = 0.05
T0 = 2.7255

# %%

camb_param = camb.set_params(
    As=As, ns=ns, H0=H0, ombh2=wb, omch2=wc, tau=tau, mnu=mnu, pivot_scalar=kp,
    TCMB=T0)
camb_interp = camb.get_matter_power_interpolator(
    camb_param, kmax=1e3, zmax=1.0, nz_step=10)

# %%

theta = np.logspace(-2, 3, 100) * u.arcmin
gamma = np.zeros(len(theta))
z_s = 0.99
z_l = 0.41

for i in range(len(theta)):
    gamma[i] = lens_magnification_bias(
        theta[i].to(u.rad).value, 2.0, z_l, z_s, camb_interp.P,
        cosmology=cosmo)

# %%

plt.title(r'$z_l = {:.2f}$, $z_s = {:.2f}$'.format(z_l, z_s))
plt.plot(theta, gamma)
plt.xscale('log')
plt.xlabel(r'Angle $\theta \, [\mathrm{arcmin}]$')
plt.ylabel(r'Bias $\Delta \gamma$')
plt.tight_layout(pad=0.3)
plt.savefig('gamma.pdf')
plt.savefig('gamma.png', dpi=300)
plt.close()

# %%

r = theta.to(u.rad).value * cosmo.comoving_distance(z_l).to(u.Mpc).value * h
sigma_crit = critical_surface_density(z_l, z_s, cosmology=cosmo, comoving=True)

plt.title(r'$z_l = {:.2f}$, $z_s = {:.2f}$'.format(z_l, z_s))
plt.plot(r, r * gamma * sigma_crit)
plt.xlim(1e-1, 1e2)
plt.xscale('log')
plt.xlabel(r'Sepration $R \, [\mathrm{Mpc}]$')
plt.ylabel(
    r'Bias $\Delta (R \times \Delta\Sigma) \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.3)
plt.savefig('delta_sigma.pdf')
plt.savefig('delta_sigma.png', dpi=300)
plt.close()

