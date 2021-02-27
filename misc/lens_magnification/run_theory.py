import camb
import zebu
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from dsigma.physics import critical_surface_density

from scipy.interpolate import interp2d
from scipy.special import jv, jn_zeros
from scipy.integrate import fixed_quad

# %%

z_d = 0.8
z_s = 0.9
alpha_d = 2.0
cosmo = zebu.cosmo

h = cosmo.H0.value / 100
pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                   omch2=(cosmo.Om0 - 0.046) * h**2)

pars.InitPower.set_params(ns=0.96, As=1e-9)
pars.set_matter_power(redshifts=np.linspace(0, z_d, 10), kmax=2000.0,
                      nonlinear=True)
results = camb.get_results(pars)

# re-calculate the result to get the correct sigma_8
sigma_8 = 0.82
As = (results.get_sigma8_0() / sigma_8)**-2 * 1e-9

pars.InitPower.set_params(As=As)
pars.set_matter_power(redshifts=np.linspace(0, z_d, 10), kmax=2000.0,
                      nonlinear=True)
results = camb.get_results(pars)

# interpolate the power spectrum
k, z, pk = results.get_matter_power_spectrum(maxkh=2000.0, npoints=1000)
pk = interp2d(k * h, z, pk / h**3, fill_value=0)

# %%


def integrand_z(z, ell):

    y = np.zeros(len(z))

    for i, z_i in enumerate(z):
        k = (ell + 0.5) / (
            (1 + z_i) * results.angular_diameter_distance(z_i))
        y[i] = (results.hubble_parameter(0) /
                results.hubble_parameter(z_i) *
                results.angular_diameter_distance2(z_i, z_d) *
                results.angular_diameter_distance2(z_i, z_s) /
                results.angular_diameter_distance(z_d) /
                results.angular_diameter_distance(z_s) *
                pk(k, z_i)[0])

    return y


def integrand_ell(ell, theta):

    return ell * jv(2, ell * theta) * np.array([
        fixed_quad(integrand_z, 0, z_d, args=(ell_i, ), n=10)[0]
        for ell_i in ell])


def gamma_t_LSS(theta, jn_zeros_max=20):

    c = 3e5
    H0 = pars.H0
    Om0 = (pars.ombh2 + pars.omch2) / (H0 / 100)**2

    return 9 * H0**3 * Om0**2 / (4 * c**3) * fixed_quad(
        integrand_ell, 0, jn_zeros(2, jn_zeros_max)[-1] / theta,
        args=(theta, ), n=jn_zeros_max*10)[0]


# %%

theta = np.logspace(-2, 3, 100) * u.arcmin
d_gamma = np.array([2 * (alpha_d - 1) * gamma_t_LSS(theta_i.to(u.rad).value)
                    for theta_i in theta])

plt.title(r'$z_l = {:.2f}$, $z_s = {:.2f}$'.format(z_d, z_s))
plt.plot(theta, d_gamma)
plt.xscale('log')
plt.xlabel(r'Angle $\theta \, [\mathrm{arcmin}]$')
plt.ylabel(r'Bias $\Delta \gamma$')
plt.tight_layout(pad=0.3)
plt.savefig('bias_gamma_theory.pdf')
plt.savefig('bias_gamma_theory.png', dpi=300)
plt.close()

# %%

r = theta.to(u.rad).value * cosmo.comoving_distance(z_d).to(u.Mpc).value * h
sigma_crit = critical_surface_density(z_d, z_s, cosmology=cosmo, comoving=True)

plt.title(r'$z_l = {:.2f}$, $z_s = {:.2f}$'.format(z_d, z_s))
plt.plot(r, r * d_gamma * sigma_crit)
plt.xlim(np.amin(zebu.rp_bins), np.amax(zebu.rp_bins))
plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(
    r'Bias $\Delta (R \times \Delta\Sigma) \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.3)
plt.savefig('bias_delta_sigma_theory.pdf')
plt.savefig('bias_delta_sigma_theory.png', dpi=300)
plt.close()
