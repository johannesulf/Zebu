import camb
import zebu
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.cosmology import Planck15
from dsigma.physics import critical_surface_density, lens_magnification_bias
from dsigma.jackknife import add_jackknife_fields
from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

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

# %%

centers = np.genfromtxt('centers.csv')
table_f = add_jackknife_fields(zebu.read_raw_data(1, 'lens', 1), centers)
table_l = add_jackknife_fields(zebu.read_raw_data(1, 'lens', 3), centers)
table_r = add_jackknife_fields(zebu.read_raw_data(1, 'random', 3), centers)

# %%

theta_bins = np.logspace(np.log10(0.01), np.log10(1.0), 11)


def foreground_lens_angular_correlation(table_f, table_l, table_r, theta_bins):

    n_f = len(table_f)
    n_l = len(table_l)
    n_r = len(table_r)

    ff = DDtheta_mocks(
        True, 4, theta_bins, table_f['ra'], table_f['dec'])['npairs']
    ll = DDtheta_mocks(
        True, 4, theta_bins, table_l['ra'], table_l['dec'])['npairs']
    rr = DDtheta_mocks(
        True, 4, theta_bins, table_r['ra'], table_r['dec'])['npairs']

    fl = DDtheta_mocks(
        False, 4, theta_bins, table_f['ra'], table_f['dec'], RA2=table_l['ra'],
        DEC2=table_l['dec'])['npairs']
    fr = DDtheta_mocks(
        False, 4, theta_bins, table_f['ra'], table_f['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    lr = DDtheta_mocks(
        False, 4, theta_bins, table_l['ra'], table_l['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']

    w_ff = convert_3d_counts_to_cf(n_f, n_f, n_r, n_r, ff, fr, fr, rr)
    w_ll = convert_3d_counts_to_cf(n_l, n_l, n_r, n_r, ll, lr, lr, rr)
    w_fl = convert_3d_counts_to_cf(n_f, n_l, n_r, n_r, fl, fr, lr, rr)

    return w_ff, w_ll, w_fl

# %%


jk_fields = np.unique(table_r['field_jk'])
n_jk = len(jk_fields)
w_ff = np.zeros((n_jk, len(theta_bins) - 1))
w_ll = np.zeros((n_jk, len(theta_bins) - 1))
w_fl = np.zeros((n_jk, len(theta_bins) - 1))

for i, jk in enumerate(jk_fields):
    print(i)
    mask_f = table_f['field_jk'] == jk
    mask_l = table_l['field_jk'] == jk
    mask_r = table_r['field_jk'] == jk
    w_ff[i], w_ll[i], w_fl[i] = foreground_lens_angular_correlation(
        table_f[~mask_f], table_l[~mask_l], table_r[~mask_r], theta_bins)

# %%

theta = np.sqrt(theta_bins[1:] * theta_bins[:-1])

plt.title(r'F: $0.3 \leq z < 0.5$, L: $0.7 \leq z < 0.9$')
plt.errorbar(theta, np.mean(w_fl, axis=0), yerr=np.std(w_fl, axis=0) *
             np.sqrt(n_jk - 1), label=r'FL')
print(np.mean(w_fl, axis=0), np.std(w_fl, axis=0) * np.sqrt(n_jk - 1))
plt.plot(theta, np.mean(w_ll, axis=0), label=r'LL')
plt.plot(theta, np.mean(w_ff, axis=0), label=r'FF')
plt.legend(loc='upper right')
plt.xlim(np.amin(theta_bins), np.amax(theta_bins))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'Angle $\theta [\mathrm{deg}]$')
plt.ylabel(r'Clustering $w(\theta)$')
plt.tight_layout(pad=0.3)
plt.savefig('clustering.pdf')
plt.savefig('clustering.png', dpi=300)
