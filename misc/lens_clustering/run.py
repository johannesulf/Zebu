import zebu
import numpy as np
import multiprocessing
from astropy import units as u
from astropy.table import Table
import matplotlib.pyplot as plt

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

# %%

theta_bins = np.logspace(np.log10(1), np.log10(100), 21) * u.arcmin


def angular_correlation_function(table_l, table_r, theta_bins):

    n_l = len(table_l)
    n_r = len(table_r)

    n = multiprocessing.cpu_count()

    ll = DDtheta_mocks(True, n, theta_bins.to(u.deg).value, table_l['ra'],
                       table_l['dec'])['npairs']
    rr = DDtheta_mocks(True, n, theta_bins.to(u.deg).value, table_r['ra'],
                       table_r['dec'])['npairs']
    lr = DDtheta_mocks(False, n, theta_bins.to(u.deg).value, table_l['ra'],
                       table_l['dec'], RA2=table_r['ra'],
                       DEC2=table_r['dec'])['npairs']

    w_ll = convert_3d_counts_to_cf(n_l, n_l, n_r, n_r, ll, lr, lr, rr)

    return w_ll

# %%


w_ll = []

for i in range(4):
    table_l = zebu.read_raw_data(1, 'lens', i)
    table_r = zebu.read_raw_data(1, 'random', i)
    w_ll.append(angular_correlation_function(table_l, table_r, theta_bins))

# %%

theta = np.sqrt(theta_bins[1:] * theta_bins[:-1]).to(u.arcmin).value

table = Table()
table['theta_min'] = theta_bins.to(u.arcmin).value[:-1]
table['theta_max'] = theta_bins.to(u.arcmin).value[1:]

for i in range(4):
    table['w_{}'.format(i + 1)] = w_ll[i]
    plt.plot(theta, w_ll[i] * theta, label='lens bin {}'.format(i + 1))

plt.xscale('log')
plt.xlabel(r'Angle $\theta [\mathrm{arcmin}]$')
plt.ylabel(r'Clustering $\theta w [\mathrm{arcmin}]$')
plt.legend(loc='upper left', frameon=False)
plt.xlim(np.amin(theta_bins.to(u.arcmin).value),
         np.amax(theta_bins.to(u.arcmin).value))
plt.xscale('log')
plt.tight_layout(pad=0.3)
plt.savefig('clustering.pdf')
plt.savefig('clustering.png', dpi=300)
plt.close()

table.write('clustering.csv')
