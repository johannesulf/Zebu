import os
import zebu
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.table import Table
from dsigma.stacking import boost_factor
from dsigma.jackknife import jackknife_resampling

# %%

survey = 'DES'
lens_bin = 2
source_bin = 3

# %%

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])
path = os.path.join(zebu.base_dir, 'stacks', 'region_1', 'precompute')
fname_base = 'l{}_s{}_{}_zspec_'.format(lens_bin, source_bin, survey.lower())

table_l = Table.read(os.path.join(path, fname_base + 'nolmag_nofib.hdf5'),
                     path='lens')
table_r = Table.read(os.path.join(path, fname_base + 'nolmag_nofib.hdf5'),
                     path='random')
b = boost_factor(table_l, table_r)
b_cov = jackknife_resampling(boost_factor, table_l, table_r=table_r)
b_err = np.sqrt(np.diag(b_cov))
plotline, caps, barlinecols = plt.errorbar(rp, b, yerr=b_err, fmt='.')
plt.setp(barlinecols[0], capstyle='round')

table_l_nomag = Table.read(
    os.path.join(path, fname_base + 'nomag_nofib.hdf5'), path='lens')
table_r_nomag = Table.read(
    os.path.join(path, fname_base + 'nomag_nofib.hdf5'), path='random')
b = boost_factor(table_l_nomag, table_r_nomag)
b_cov = jackknife_resampling(
    boost_factor, table_l_nomag, table_r=table_r_nomag)
b_err = np.sqrt(np.diag(b_cov))
plotline, caps, barlinecols = plt.errorbar(rp * 1.06, b, yerr=b_err, fmt='.')
plt.setp(barlinecols[0], capstyle='round')

plt.xscale('log')
plt.axhline(1.0, color='black', ls='--')

plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Boost Estimate $\hat{b}$')
plt.tight_layout(pad=0.3)
plt.savefig('boost_zspec_mag_vs_nomag.pdf')
plt.savefig('boost_zspec_mag_vs_nomag.png', dpi=300)
plt.close()

# %%

X = np.ones(len(b))
X = X.T
b_pre = np.linalg.inv(b_cov)
b_mean_var = 1 / np.dot(np.dot(X.T, b_pre), X)
b_mean = np.dot(np.dot(b_mean_var * X.T,  b_pre), b)
print('Average boost w/o magn.: {:.4f}+/-{:.4f}'.format(
    b_mean, np.sqrt(b_mean_var)))

# %%


def boost_factor_diff(table_l, table_r=None, table_l_2=None, table_r_2=None):
    return (boost_factor(table_l, table_r) -
            boost_factor(table_l_2, table_r_2))


b = boost_factor_diff(table_l, table_r, table_l_nomag, table_r_nomag)
b_cov = jackknife_resampling(
    boost_factor_diff, table_l, table_r=table_r,
    table_l_2=table_l_nomag, table_r_2=table_r_nomag)
b_err = np.sqrt(np.diag(b_cov))
plotline, caps, barlinecols = plt.errorbar(
    rp, b * 100, yerr=b_err * 100, fmt='.', color='royalblue')
plt.setp(barlinecols[0], capstyle='round')

plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(
    lambda y, p: r'{:+g}\%'.format(y) if y != 0 else r'0\%'))
plt.xscale('log')
plt.axhline(0, color='black', ls='--')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Boost Estimate Bias $\Delta \hat{b}$')
plt.title(r'{}, ${:.1f} \leq z_l < {:.1f}$, ${:.2f} \leq z_s < {:.2f}$'.format(
    survey, zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1],
    zebu.source_z_bins[survey.lower()][source_bin],
    zebu.source_z_bins[survey.lower()][source_bin + 1]))
plt.tight_layout(pad=0.3)
plt.savefig('boost_bias.pdf')
plt.savefig('boost_bias.png', dpi=300)
plt.close()

# %%

fname_base = 'l{}_s{}_{}_'.format(lens_bin, source_bin, survey.lower())

table_l = Table.read(os.path.join(path, fname_base + 'nomag_nofib.hdf5'),
                     path='lens')
table_r = Table.read(os.path.join(path, fname_base + 'nomag_nofib.hdf5'),
                     path='random')
b = boost_factor(table_l, table_r)
b_cov = jackknife_resampling(boost_factor, table_l, table_r=table_r)
b_err = np.sqrt(np.diag(b_cov))
plotline, caps, barlinecols = plt.errorbar(
    rp, b, yerr=b_err, fmt='.', label='w/o source magnification')
plt.setp(barlinecols[0], capstyle='round')

table_l = Table.read(os.path.join(path, fname_base + 'nolmag_nofib.hdf5'),
                     path='lens')
table_r = Table.read(os.path.join(path, fname_base + 'nolmag_nofib.hdf5'),
                     path='random')
b = boost_factor(table_l, table_r)
b_cov = jackknife_resampling(boost_factor, table_l, table_r=table_r)
b_err = np.sqrt(np.diag(b_cov))
plotline, caps, barlinecols = plt.errorbar(
    rp * 1.05, b, yerr=b_err, fmt='.', label='with source magnification')
plt.setp(barlinecols[0], capstyle='round')

plt.legend(loc='best')
plt.xscale('log')
plt.axhline(1, color='black', ls='--')
plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Boost Estimate $\hat{b}$')
plt.title(r'{}, ${:.1f} \leq z_l < {:.1f}$, ${:.2f} \leq z_s < {:.2f}$'.format(
    survey, zebu.lens_z_bins[lens_bin], zebu.lens_z_bins[lens_bin + 1],
    zebu.source_z_bins[survey.lower()][source_bin],
    zebu.source_z_bins[survey.lower()][source_bin + 1]))
plt.tight_layout(pad=0.3)
plt.savefig('boost_photo.pdf')
plt.savefig('boost_photo.png', dpi=300)
plt.close()
