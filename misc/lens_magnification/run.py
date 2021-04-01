import os
import camb
import zebu
import numpy as np
import matplotlib.pyplot as plt
from dsigma.physics import lens_magnification_shear_bias
from dsigma.stacking import excess_surface_density, lens_magnification_bias
from astropy.table import Table

# %%

cmap = plt.get_cmap('viridis')

for i in range(4):

    color = cmap(i / 3)

    table_m = Table.read(
        os.path.join(zebu.base_dir, 'mocks', 'magnification.hdf5'),
        path='lens_{}'.format(i))

    mu = np.array(table_m.meta['mu'])

    f = (table_m['n'].T / table_m['n'][:, np.argmin(np.abs(mu - 1))]).T

    f_mean = np.mean(f, axis=0)
    alpha = (f_mean[1] - f_mean[3]) / (mu[1] - mu[3])

    plt.errorbar(mu, np.mean(f, axis=0),
                 yerr=np.std(f, axis=0, ddof=1) / np.sqrt(f.shape[0]) * 10,
                 fmt='x', color=color,
                 label=r'lens bin {}: $\alpha = {:.2f}$'.format(i + 1, alpha))

    plt.plot(mu, 1 + (mu - 1) * alpha, ls='--', color=color)

plt.legend(loc='best')
plt.xlabel(r'$\mu$')
plt.ylabel(r'$\mu \times n(\mu) / n(\mu = 1)$')
plt.tight_layout(pad=0.3)
plt.savefig('flux_magnification.pdf')
plt.savefig('flux_magnification.png', dpi=300)
plt.close()

# %%

z_d = 0.41
z_s = 0.99
alpha_d = 2.71

h = 0.73
pars = camb.CAMBparams()
pars.set_cosmology(H0=100 * h, ombh2=0.045 * h**2, omch2=0.205 * h**2)
pars.InitPower.set_params(ns=1.0, As=2.83e-9)
pars.set_matter_power(redshifts=np.linspace(z_d, 0, 10), kmax=2000.0,
                      nonlinear=True)

camb_results = camb.get_results(pars)

# %%

theta = np.deg2rad(np.logspace(-1, 1.5, 20) / 60)
d_gamma = [lens_magnification_shear_bias(theta_i, alpha_d, z_d, z_s,
                                         camb_results) for theta_i in theta]

plt.plot(np.rad2deg(theta) * 60, 1e5 * np.array(d_gamma))

plt.title(r'$z_d = {:.2f}, z_s = {:.2f}, \alpha_d = {:.2f}$'.format(
    z_d, z_s, alpha_d))
plt.xscale('log')
plt.xlabel(r'$\theta$ in arcmin')
plt.ylabel(r'Bias $10^5 \times \Delta \gamma_t$')
plt.xlim(0.5, 20)
plt.tight_layout(pad=0.3)
plt.savefig('unruh_20.pdf')
plt.savefig('unruh_20.png', dpi=300)
plt.close()

# %%

lens_bin = 2
camb_results = zebu.get_camb_results()
rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

# %%

for i in range(3):

    source_bin = 3 - i

    path = os.path.join(
        zebu.base_dir, 'stacks', 'region_1', 'precompute',
        'l{}_s{}_gen_zspec_nosmag.hdf5'.format(lens_bin, source_bin))

    table_l = Table.read(path, path='lens')
    table_r = Table.read(path, path='random')

    kwargs = zebu.stacking_kwargs('gen')

    if source_bin == 3:
        ds_ref = excess_surface_density(table_l, table_r=table_r, **kwargs)

    ds_lm = lens_magnification_bias(
        table_l, zebu.alpha_l[lens_bin], camb_results,
        photo_z_dilution_correction=True)

    plt.plot(rp, rp * ds_lm, label=r'${:.1f} \leq z_s < {:.1f}$'.format(
        zebu.source_z_bins['gen'][source_bin],
        zebu.source_z_bins['gen'][source_bin + 1]))

plt.title(r'${:.1f} \leq z_l < {:.1f}$'.format(
    zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1]))
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(
    r'Bias $\Delta (R \times \Delta\Sigma) \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.3)
plt.savefig('bias_delta_sigma_theory.pdf')
plt.savefig('bias_delta_sigma_theory.png', dpi=300)
plt.close()
