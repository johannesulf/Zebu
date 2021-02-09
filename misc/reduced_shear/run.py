import zebu
import numpy as np
from astropy import units as u
from astropy import constants
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from colossus.halo.concentration import concentration
from colossus.lss.mass_function import massFunction
from scipy.integrate import fixed_quad
from scipy.special import erf


def estimate_convergence(rp, gamma, deg=6, reduced_shear=False):

    if reduced_shear:
        g = gamma
        for i in range(10):
            kappa = estimate_convergence(rp, gamma, deg=deg)
            gamma = g * (1 - kappa)
        return kappa
    else:
        p = np.polyfit(np.log(rp), np.log(gamma), deg)
        gamma = lambda rp_bar: np.exp(np.polyval(p, np.log(rp_bar)))
        integrand = lambda rp_bar: 2 * gamma(rp_bar) / rp_bar

        kappa = np.array([fixed_quad(integrand, rp_i, np.amax(rp), n=100)[0] -
                          gamma(rp_i) for rp_i in rp])
        kappa = kappa - np.amin(kappa)

    return kappa


cosmo = cosmology.setCosmology('planck15')

z_l = 0.5
z_s = 1.0

log_mvir = np.linspace(10.0, 15.0, 100)
dn_dln_mvir = massFunction(10**log_mvir, z_l, q_out='dndlnM')
dn = dn_dln_mvir * np.diff(np.log(10**log_mvir))[0]

log_mmin = 13.2
sigma_logm = 0.1
n_gal = 0.5 * (1 + erf((log_mvir - log_mmin) / sigma_logm))

dn *= n_gal

print('Number density: {:.1e} h^3 Mpc^-3'.format(np.sum(dn)))

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

gamma = np.zeros_like(rp)
g = np.zeros_like(rp)
kappa = np.zeros_like(rp)

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

        kappa_j = profile.surfaceDensity(rp_phys) / sigma_crit
        gamma_j = profile.deltaSigma(rp_phys) / sigma_crit

        gamma[j] += w * gamma_j
        g[j] += w * gamma_j / (1 - kappa_j)
        kappa[j] += w * kappa_j

# %%

plt.plot(rp, g / gamma, color='black')
plt.plot(rp, 1 / (1 - kappa), color='black', ls='--', zorder=0)
plt.plot(rp, 1 / (1 - estimate_convergence(rp, g, deg=9, reduced_shear=True)),
         ls=':', color='grey', zorder=1)
plt.xscale('log')

plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Shear Ratio $\langle g \rangle / \langle \gamma \rangle$')
plt.title(r'$z_l = {:.1f}, z_s = {:.1f},'.format(z_l, z_s) +
          r'n_{\rm gal} = ' + '{:.1f}'.format(1e4 * np.sum(dn)) +
          r'\times 10^{-4}$')
plt.tight_layout(pad=0.3)
plt.savefig('reduced_shear.pdf')
plt.savefig('reduced_shear.png', dpi=300)
plt.close()

# %%

kappa_est = estimate_convergence(rp, g, deg=9, reduced_shear=True)

plt.plot(rp, g * (1 - kappa_est) / gamma, color='black')

plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Ratio $\hat{\gamma} / \gamma$')
plt.title(r'$z_l = {:.1f}, z_s = {:.1f},'.format(z_l, z_s) +
          r'n_{\rm gal} = ' + '{:.1f}'.format(1e4 * np.sum(dn)) +
          r'\times 10^{-4}$')
plt.tight_layout(pad=0.3)
plt.savefig('shear_recovery.pdf')
plt.savefig('shear_recovery.png', dpi=300)
