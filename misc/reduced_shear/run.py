import zebu
import numpy as np
from astropy import units as u
from astropy import constants
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from colossus.halo.concentration import concentration
from colossus.lss.mass_function import massFunction

cosmo = cosmology.setCosmology('planck15')

z_l = 0.5
z_s = 1.0
n_gal = 2e-4

log_mvir = np.linspace(10.0, 15.0, 100)
dn_dln_mvir = massFunction(10**log_mvir, z_l, q_out='dndlnM')
dn = dn_dln_mvir * np.diff(np.log(10**log_mvir))[0]

# Cut lower masses such that total number density is n_gal.
dn = np.minimum(
    dn[::-1], n_gal - np.concatenate([[0], np.cumsum(dn[::-1])[:-1]]))[::-1]
dn = np.where(dn < 0, 0, dn)

# %%

rp = np.logspace(np.log10(np.amin(zebu.rp_bins)),
                 np.log10(np.amax(zebu.rp_bins)), 100)

dc_l = cosmo.comovingDistance(z_max=z_l) * u.Mpc
dc_s = cosmo.comovingDistance(z_max=z_s) * u.Mpc

d_l = dc_l / (1 + z_l)
d_s = dc_s / (1 + z_s)
d_ls = ((dc_s - dc_l) / (1 + z_s))

sigma_crit = (constants.c**2 / (4 * np.pi * constants.G) * d_s / (
    d_l * d_ls)).to(u.solMass / u.kpc**2).value

shear = np.zeros_like(rp)
red_shear = np.zeros_like(rp)
magn = np.zeros_like(rp)

# %%

for i, mvir in enumerate(10**log_mvir):

    w = dn[i] / np.sum(dn)
    if w == 0:
        continue

    profile = profile_nfw.NFWProfile(
        M=mvir, c=concentration(mvir, 'vir', z_l, model='diemer19'), z=z_l,
        mdef='vir')

    for j in range(len(rp)):

        rp_phys = 1000 * rp[j] / (1 + z_l)  # phys. in kpc/h

        kappa = profile.surfaceDensity(rp_phys) / sigma_crit
        gamma = profile.deltaSigma(rp_phys) / sigma_crit

        shear[j] += w * gamma
        red_shear[j] += w * gamma / (1 - kappa)
        magn[j] += w * kappa

# %%

plt.plot(rp, red_shear / shear, color='black',
         label=r'$\langle \gamma / (1 - \kappa) \rangle / \langle \gamma \rangle$')
plt.plot(rp, 1 / (1 - magn), color='black', ls='--',
         label=r'$1 / (1 - \langle \kappa \rangle)$')
plt.xscale('log')

plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Shear Ratios')
plt.legend(loc='best')
plt.title(r'$z_l = {:.1f}, z_s = {:.1f},'.format(z_l, z_s) +
          r'n_{\rm gal} = ' + '{:.1f}'.format(1e4 * n_gal) +
          r'\, \times 10^{-4} h^3 \, \mathrm{Mpc}^{-3}$')
plt.tight_layout(pad=0.3)
plt.savefig('reduced_shear.pdf')
plt.savefig('reduced_shear.png', dpi=300)
