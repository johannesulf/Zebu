import os
import camb
import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from scipy.spatial import cKDTree


base_dir = os.path.dirname(os.path.abspath(__file__))
cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
rp_bins = 0.2 * np.logspace(0, 2, 21)
theta_bins = 3 * np.logspace(0, 2, 21)

lens_z_bins = np.linspace(0.1, 0.9, 5)
source_z_bins = {
    'gen': np.array([0.5, 0.7, 0.9, 1.1, 1.5]),
    'des': np.array([0.2, 0.43, 0.63, 0.9, 1.3]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}

alpha_l = [0.818, 1.658, 2.568, 1.922]

MAP_IA = None
RED_IA = None


def stacking_kwargs(survey):

    if survey.lower() == 'gen':
        return {'photo_z_dilution_correction': True,
                'boost_correction': False, 'random_subtraction': False}
    elif survey.lower() in ['des', 'hsc', 'kids']:
        return {'photo_z_dilution_correction': True,
                'boost_correction': False, 'random_subtraction': False,
                'scalar_shear_response_correction': survey.lower() != 'des',
                'matrix_shear_response_correction': survey.lower() == 'des',
                'shear_responsivity_correction': survey.lower() == 'hsc'}
    else:
        raise RuntimeError('Unkown lensing survey {}.'.format(survey))


def read_mock_data(catalog_type, z_bin, survey='gen', magnification=False,
                   fiber_assignment=False, unlensed_coordinates=False,
                   intrinsic_alignment=False):

    if z_bin == 'all':
        return vstack([read_mock_data(
            catalog_type, source_bin, survey=survey,
            magnification=magnification, fiber_assignment=fiber_assignment,
            unlensed_coordinates=unlensed_coordinates,
            intrinsic_alignment=intrinsic_alignment)
                       for source_bin in range(5 if survey.lower() == 'kids'
                                               else 4)])

    if catalog_type not in ['source', 'lens', 'calibration', 'random']:
        raise RuntimeError('Unkown catalog type: {}.'.format(catalog_type))

    if catalog_type in ['source', 'calibration']:
        fname = '{}{}'.format(catalog_type[0], z_bin)
        fname = fname + '_{}'.format(survey.lower())
    else:
        fname = 'bgs' if z_bin in [0, 1] else 'lrg'
        if catalog_type == 'random':
            fname += '_rand'

    if not fiber_assignment and catalog_type == 'lens':
        fname = fname + '_nofib'

    if not magnification and catalog_type != 'random':
        fname = fname + '_nomag'

    fname = fname + '.hdf5'

    table = vstack([Table.read(os.path.join(
        base_dir, 'mocks', 'mocks_' + subsample, fname)) for subsample
        in 'abcdefghij'], metadata_conflicts='silent')

    if catalog_type == 'calibration':
        if survey.lower() == 'des':
            table['w_sys'] = 0.5 * (table['R_11'] + table['R_22'])
        elif survey.lower() == 'hsc':
            table['w_sys'] = (1 - table['e_rms']**2) * (1 + table['m'])
        else:
            table['w_sys'] = 1.0

    if catalog_type == 'random':
        table['w_sys'] = 1.0

    if catalog_type == 'source':
        table['e_2'] = - table['e_2']
        table['g_2'] = - table['g_2']

    if catalog_type in ['lens', 'random']:
        table = table[(table['z'] >= lens_z_bins[z_bin]) &
                      (table['z'] < lens_z_bins[z_bin + 1])]

    for key in table.colnames:
        table[key] = table[key].astype(np.float64)

    if unlensed_coordinates:
        table['ra'] = table['ra_true']
        table['dec'] = table['dec_true']

    if catalog_type == 'source' and intrinsic_alignment:
        global MAP_IA
        global RED_IA
        path = os.path.join(base_dir, 'mocks')
        if MAP_IA is None:
            MAP_IA = Table.read(os.path.join(path, 'ia.hdf5'), path='ia')
            RED_IA = Table.read(os.path.join(path, 'ia.hdf5'), path='z')
        for i in range(len(RED_IA)):
            select = ((table['z_true'] >= RED_IA['z_min'][i]) &
                      (table['z_true'] < RED_IA['z_max'][i]))
            pix = hp.ang2pix(
                2048, table['ra'][select], table['dec'][select],
                lonlat=True)
            row = np.searchsorted(MAP_IA['pix'], pix)
            table['e_1'][select] += MAP_IA['e_1'][:, i][row]
            table['e_2'][select] -= MAP_IA['e_2'][:, i][row]
            table['g_1'][select] += MAP_IA['e_1'][:, i][row]
            table['g_2'][select] -= MAP_IA['e_2'][:, i][row]

    return table


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


def get_camb_results():

    h = cosmo.H0.value / 100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                       omch2=(cosmo.Om0 - 0.046) * h**2)
    pars.InitPower.set_params(ns=0.96, As=9.93075e-10)
    pars.set_matter_power(redshifts=np.linspace(2, 0, 20), kmax=2000.0,
                          nonlinear=True)

    return camb.get_results(pars)
