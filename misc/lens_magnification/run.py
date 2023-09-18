import camb
import zebu
import numpy as np
import matplotlib.pyplot as plt
from dsigma.physics import lens_magnification_shear_bias

# %%

alpha = []

for survey in ['bgs', 'lrg']:

    table_l_mag = zebu.read_mock_catalog(
        survey, zebu.MOCK_PATH / 'buzzard-4', zebu.PIXELS, magnification=True)
    table_l_nomag = zebu.read_mock_catalog(
        survey, zebu.MOCK_PATH / 'buzzard-4', zebu.PIXELS, magnification=False)

    for lens_bin, (z_min, z_max) in enumerate(zip(
            zebu.LENS_Z_BINS[survey][:-1], zebu.LENS_Z_BINS[survey][1:])):

        if survey == 'bgs':
            table_l_mag = table_l_mag[
                table_l_mag['abs_mag_r'] < zebu.ABS_MAG_R_MAX[lens_bin]]
            table_l_nomag = table_l_nomag[
                table_l_nomag['abs_mag_r'] < zebu.ABS_MAG_R_MAX[lens_bin]]

        select_mag = (z_min < table_l_mag['z']) & (table_l_mag['z'] < z_max)
        select_nomag = (z_min < table_l_nomag['z']) & (
            table_l_nomag['z'] < z_max)
        mu_min = np.percentile(table_l_nomag[select_nomag]['mu'], 20)
        mu_max = np.percentile(table_l_nomag[select_nomag]['mu'], 80)
        mu_bins = bins = np.linspace(mu_min, mu_max, 101)
        mu = 0.5 * (mu_bins[1:] + mu_bins[:-1])
        h_mag = np.histogram(table_l_mag['mu'][select_mag], bins=mu_bins)[0]
        h_nomag = np.histogram(
            table_l_nomag['mu'][select_nomag], bins=mu_bins)[0]
        alpha.append(np.corrcoef(
            mu, h_mag / h_nomag)[0][1] * np.std(h_mag / h_nomag) / np.std(mu))

print('alpha values: {}'.format(alpha))

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
