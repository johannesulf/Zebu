import zebu
import numpy as np
from astropy import units as u
from astropy import constants
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw
from colossus.halo.concentration import concentration
from colossus.lss.mass_function import massFunction
from scipy.special import erf

# %%

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
                 np.log10(np.amax(zebu.rp_bins)), 200)

dc_l = cosmo.comovingDistance(z_max=z_l) * u.Mpc
dc_s = cosmo.comovingDistance(z_max=z_s) * u.Mpc

d_l = dc_l / (1 + z_l)
d_s = dc_s / (1 + z_s)
d_ls = ((dc_s - dc_l) / (1 + z_s))

sigma_crit = (constants.c**2 / (4 * np.pi * constants.G) * d_s / (
    d_l * d_ls)).to(u.solMass / u.kpc**2).value

ds = np.zeros((len(log_mvir), len(rp)))
kappa = np.zeros((len(log_mvir), len(rp)))

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

        ds[i, j] = profile.surfaceDensity(rp_phys) / (1 + z_l)**2
        kappa[i, j] = profile.surfaceDensity(rp_phys) / sigma_crit

# %%

alpha_s = 2.0

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

ds_nomag = np.average(ds, axis=0, weights=dn)
ax1.plot(rp, rp * ds_nomag / 1e6, label='w/o source magn.')

ds_mag = np.average(
    ds, axis=0, weights=dn[:, np.newaxis] * (1 + 2 * (alpha_s - 1) * kappa))
ax1.plot(rp, rp * ds_mag / 1e6, label='with source magn.')
ax1.legend(loc='best')

ax2.plot(rp, 100 * (ds_mag - ds_nomag) / ds_nomag, color='black')

ax1.set_xscale('log')
ax2.set_xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
ax1.set_ylabel(r'$r_p \Delta\Sigma [10^6 M_\odot / pc]$')
ax2.set_ylabel(r'Difference [\%]')

plt.tight_layout(pad=0.3)
plt.subplots_adjust(hspace=0)
plt.savefig('1halo_source_magnification.pdf')
plt.savefig('1halo_source_magnification.png', dpi=300)
