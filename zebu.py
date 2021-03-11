import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree
from dsigma.stacking import excess_surface_density

base_dir = os.path.dirname(os.path.abspath(__file__))
cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
rp_bins = 0.2 * np.logspace(0, 2, 21)


def stacking_kwargs(survey):

    if survey.lower() == 'gen':
        return {'photo_z_dilution_correction': True,
                'boost_correction': True, 'random_subtraction': True}
    elif survey.lower() in ['des', 'hsc', 'kids']:
        return {'photo_z_dilution_correction': True,
                'boost_correction': False, 'random_subtraction': True,
                'shear_bias_correction': survey.lower() != 'des',
                'shear_responsivity_correction': survey.lower() == 'hsc',
                'metacalibration_response_correction': survey.lower() == 'des'}
    else:
        raise RuntimeError('Unkown lensing survey {}.'.format(survey))


lens_z_bins = np.linspace(0.1, 0.9, 5)
source_z_bins = {
    'gen': [0.5, 0.7, 0.9, 1.1, 1.5],
    'des': [0.2, 0.43, 0.63, 0.9, 1.3],
    'hsc': [0.3, 0.6, 0.9, 1.2, 1.5],
    'kids': [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]}


def read_mock_data(catalog_type, z_bin, survey='gen', region=1,
                   magnification=True, fiber_assignment=False):

    if catalog_type not in ['source', 'lens', 'calibration', 'random']:
        raise RuntimeError('Unkown catalog type: {}.'.format(catalog_type))

    path = os.path.join(base_dir, 'mocks', 'region_{}'.format(region))

    fname = '{}{}'.format(catalog_type[0], z_bin)

    if catalog_type in ['source', 'calibration']:
        fname = fname + '_{}'.format(survey.lower())

    if not fiber_assignment and catalog_type == 'lens':
        fname = fname + '_nofib'

    if not magnification and catalog_type != 'random':
        fname = fname + '_nomag'

    fname = fname + '.hdf5'

    table = Table.read(os.path.join(path, fname))

    if catalog_type in ['calibration', 'source'] and survey == 'gen':
        table['w'] = 1.0

    if catalog_type == 'calibration':
        table['w_sys'] = 1.0

    if catalog_type == 'lens' and not fiber_assignment:
        table['w_sys'] = 1.0

    if catalog_type == 'random':
        table['w_sys'] = 1.0

    return table


def ds_diff(table_l, table_r=None, table_l_2=None, table_r_2=None,
            survey_1=None, survey_2=None, ds_norm=None, stage=0):

    for survey in [survey_1, survey_2]:
        if survey not in ['gen', 'hsc', 'kids', 'des']:
            raise RuntimeError('Unkown survey!')

    ds_1 = excess_surface_density(table_l, table_r=table_r,
                                  **stacking_kwargs(survey_1))
    ds_2 = excess_surface_density(table_l_2, table_r=table_r_2,
                                  **stacking_kwargs(survey_2))

    if ds_norm is not None:
        return (ds_1 - ds_2) / ds_norm

    return ds_1 - ds_2


def linear_regression(x, y, cov, fit_constant=True, return_err=False):

    if not fit_constant:
        X = np.vstack([np.ones_like(x), x]).T
    else:
        X = np.atleast_2d(np.ones_like(x)).T

    pre = np.linalg.inv(cov)
    beta_cov = np.linalg.inv(np.dot(np.dot(X.T, pre), X))
    beta = np.dot(np.dot(np.dot(beta_cov, X.T),  pre), y)

    if return_err:
        return beta, beta_cov

    return beta


def project_onto_som(som, pos):

    weights = som._weights.reshape(
        np.prod(som._weights.shape[:-1]), som._weights.shape[-1])

    kdtree = cKDTree(weights)

    d3d, idx = kdtree.query(pos)
    idx_x = idx // som._weights.shape[1]
    idx_y = idx % som._weights.shape[1]

    return d3d, idx_x, idx_y


def som_f_of_x(som, pos, x, f=np.mean):

    d3d, idx_x, idx_y = project_onto_som(som, pos)

    map_shape = som._weights.shape[:-1]

    idx = idx_x * map_shape[1] + idx_y

    y = np.zeros(np.prod(map_shape))

    x = x[np.argsort(idx)]
    idx = np.sort(idx)

    for i in range(len(y)):
        i_min = np.searchsorted(idx, i)
        i_max = np.searchsorted(idx, i + 1)
        if i_max - i_min > 0:
            y[i] = f(x[i_min:i_max].astype(np.float64))
        else:
            y[i] = np.nan

    return y.reshape(map_shape)
