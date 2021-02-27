import zebu
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from scipy.interpolate import interp1d

from dsigma.physics import critical_surface_density
from dsigma.jackknife import add_jackknife_fields
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog
from dsigma.stacking import raw_excess_surface_density

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

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

    n = multiprocessing.cpu_count()

    fl = DDtheta_mocks(
        False, n, theta_bins, table_f['ra'], table_f['dec'], RA2=table_l['ra'],
        DEC2=table_l['dec'])['npairs']
    fr = DDtheta_mocks(
        False, n, theta_bins, table_f['ra'], table_f['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    lr = DDtheta_mocks(
        False, n, theta_bins, table_l['ra'], table_l['dec'], RA2=table_r['ra'],
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
plt.close()

# %%

table_f = vstack([zebu.read_raw_data(1, 'lens', i) for i in range(3)])

# %%


def angular_correlation_coefficient(table_1, table_2, table_r, theta_bins):

    n_1 = len(table_1)
    n_2 = len(table_2)
    n_r = len(table_r)

    d1d1 = DDtheta_mocks(
        True, 4, theta_bins, table_1['ra'], table_1['dec'])['npairs']
    rr = DDtheta_mocks(
        True, 4, theta_bins, table_r['ra'], table_r['dec'])['npairs']

    n = multiprocessing.cpu_count()

    d1r = DDtheta_mocks(
        False, n, theta_bins, table_1['ra'], table_1['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    d2r = DDtheta_mocks(
        False, n, theta_bins, table_2['ra'], table_2['dec'], RA2=table_r['ra'],
        DEC2=table_r['dec'])['npairs']
    d1d2 = DDtheta_mocks(
        False, n, theta_bins, table_1['ra'], table_1['dec'], RA2=table_2['ra'],
        DEC2=table_2['dec'])['npairs']

    w_11 = convert_3d_counts_to_cf(n_1, n_1, n_r, n_r, d1d1, d1r, d1r, rr)
    w_12 = convert_3d_counts_to_cf(n_1, n_2, n_r, n_r, d1d2, d1r, d2r, rr)

    return np.average(w_12 / w_11, weights=np.diff(theta_bins**2)**-0.5)

# %%


z_bins = np.linspace(0.1, 0.7, 7)
theta_bins = np.logspace(-2, -1, 3)

k_l = np.zeros(len(z_bins) - 1)
k = np.zeros((len(z_bins) - 1, len(z_bins) - 1))

for i in range(len(k_l)):
    use_i = (z_bins[i] <= table_f['z']) & (table_f['z'] < z_bins[i + 1])
    k_l[i] = angular_correlation_coefficient(table_f[use_i], table_l, table_r,
                                             theta_bins)

    for j in range(len(k_l)):
        print(i, j)

        if i == j:
            k[i, j] = 1
            continue

        use_j = (z_bins[j] <= table_f['z']) & (table_f['z'] < z_bins[j + 1])
        k[i, j] = angular_correlation_coefficient(
            table_f[use_i], table_f[use_j], table_r, theta_bins)

# %%

z = 0.5 * (z_bins[1:] + z_bins[:-1])

w_f = np.array([
    2 * k_l[i] - np.sum((k[i, :] * k_l)) for i in range(len(k_l))])

plt.plot(z, w_f)
plt.xlabel(r'Foreground Redshift $z_F$')
plt.ylabel(r'Foreground Weight $w_F$')
plt.axhline(0, color='black', ls='--')
plt.tight_layout(pad=0.3)
plt.savefig('w_f.pdf')
plt.savefig('w_f.png', dpi=300)
plt.close()

# %%

w_f_tot = np.sum(w_f)

for i in range(len(z_bins) - 1):
    use = (z_bins[i] <= table_f['z']) & (table_f['z'] < z_bins[i + 1])
    table_f['w_sys'] = np.where(use, w_f[i] / w_f_tot, table_f['w_sys'])

table_f['z'] = np.random.choice(table_l['z'], size=len(table_f))

# %%

table_c = zebu.read_raw_data(1, 'calibration', 4, survey='kids')
table_c = add_maximum_lens_redshift(table_c, dz_min=0.2, z_err_factor=0)
z_lens = np.linspace(np.amin(table_f['z']), np.amax(table_f['z']), 1000)

w_sys = np.zeros_like(z_lens)

for i in range(len(z_lens)):
    sigma_crit = critical_surface_density(z_lens[i], table_c['z'],
                                          cosmology=zebu.cosmo)
    w_sys[i] = np.sum(table_c['w'] * table_c['w_sys'] * sigma_crit**-2 *
                      (z_lens[i] < table_c['z_l_max']))**-1

w_sys = w_sys / np.amax(w_sys)
table_f['w_sys'] *= interp1d(z_lens, w_sys)(table_f['z'])

# %%

table_s = zebu.read_raw_data(1, 'source', 4, survey='kids')
table_s['e_1'] = table_s['g_1']
table_s['e_2'] = table_s['g_2']
table_s = add_maximum_lens_redshift(table_s, dz_min=0.2, z_err_factor=0)

table_f = precompute_catalog(
    table_f[::10], table_s, zebu.rp_bins, cosmology=zebu.cosmo,
    n_jobs=multiprocessing.cpu_count())

# %%

result = Table.read('result_3_4_gamma.csv')
ds_f = raw_excess_surface_density(table_f)
ds_f = w_f_tot * (ds_f - result['ds_r']) * result['f_bias'] / result['1 + m']

# %%

plt.errorbar(result['rp'], result['rp'] * result['ds'],
             yerr=result['rp'] * result['ds_err'], label='default')
plt.errorbar(result['rp'], result['rp'] * (result['ds'] - ds_f),
             label='self-calibrated')
plt.xscale('log')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'$r_p \Delta \Sigma \, [10^6 M_\odot / \mathrm{pc}]$')
plt.legend(loc='upper left')
plt.tight_layout(pad=0.3)
plt.savefig('delta_sigma_calibration.pdf')
plt.savefig('delta_sigma_calibration.png', dpi=300)
plt.close()

# %%

plt.plot(result['rp'], result['rp'] * ds_f)
plt.xscale('log')

plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(
    r'Bias $\Delta (R \times \Delta\Sigma) \, [10^6 M_\odot / \mathrm{pc}]$')
plt.tight_layout(pad=0.3)
plt.savefig('bias_delta_sigma_mocks.pdf')
plt.savefig('bias_delta_sigma_mocks.png', dpi=300)
plt.close()
