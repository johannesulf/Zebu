import matplotlib.pyplot as plt
import numpy as np
import zebu

from astropy.table import Table

name = 'lens_magnification'
label = 'residual fibre assignment'
survey = 'des'
statistic = 'ds'

data = Table.read(
    zebu.BASE_PATH / 'stacks' / 'plots_absolute' / '{}_{}_{}.csv'.format(
        name, statistic, survey))
cov = zebu.covariance(statistic, survey)[0]

# %%

use = (data['lens_bin'] <= 4) & (data['r'] > 0) & (
    data['lens_bin'] >= 4) & (data['source_bin'] <= 0)
bias = data['value']

# %%


def significance(bias, cov, use=None):

    if use is None:
        use = np.ones(len(bias), dtype=bool)

    bias = bias[use]
    cov = cov[np.outer(use, use)].reshape(np.sum(use), np.sum(use))
    err = np.sqrt(np.diag(cov))

    bias = bias / err
    cov = cov / np.outer(err, err)
    print(bias)

    pre = np.linalg.inv(cov)

    return np.dot(np.dot(bias, pre), bias)
