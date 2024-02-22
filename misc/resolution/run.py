import numpy as np
from analytic_model import Model
import matplotlib.pyplot as plt

# This code uses a modified version of the halo model in Lange et al. (2019).
# The only difference is that the one-halo correlation function uses a fixed
# k_max whereas the original code kept r_halo x k_max fixed to scale k_max
# with the halo size. The fixed value is k_times_r_halo_max in the
# configuration file.
for resolution, label in zip(
        ['normal', 'kmax2'],
        ['normal', r'$k_{\rm max} = 2 h \, \mathrm{Mpc}^{-1}$']):

    model = Model(resolution)

    z = 0.15
    cosmo = {'flat': True, 'H0': 67.74, 'Om0': 0.3089, 'Ob0': 0.0486,
             'sigma8': 0.8159, 'ns': 0.9667}

    model.cache_cosmology(cosmo, z)

    nuisance_parameters = {'psi': 0.9, 'eta': 0.07, 'r_cen': 0.01,
                           'r_sat': 1.0}
    model.set_nuisance_parameters(nuisance_parameters)

    rp_bins = np.logspace(-2, 2, 21)
    pi_max = 100
    r_ds_bins = np.logspace(-2, 2, 21)
    model.cache_correlation_functions(rp_bins, pi_max, r_ds_bins)

    hod_param_dict = dict(log_m_min=13.096, sigma_log_m=0.39, alpha=1.18,
                          log_m_0=11.67, log_m_1=14.244, f_compl=0.83)

    model.set_galaxy_halo_connection(hod_param_dict)
    esd = model.excess_surface_density()

    r = np.sqrt(r_ds_bins[1:] * r_ds_bins[:-1])
    plt.plot(r, r * esd, label=label)

plt.xscale('log')
plt.xlabel(r'$r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \Delta\Sigma(r_p) \, [10^6 M_\odot / \mathrm{pc}]$')
plt.legend(loc='best', frameon=False)
plt.tight_layout(pad=0.3)
plt.savefig('kmax.pdf')
plt.savefig('kmax.png', dpi=300)
plt.close()
