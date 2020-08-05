import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
rp_bins = 0.05 * np.logspace(0, 3, 31)

if '/data/groups/leauthaud/' in os.getcwd():
    host = 'lux'
elif '/project/projectdirs/desi/' in os.getcwd():
    host = 'cori'
else:
    host = 'personal'


def check_stage(stage):
    if not isinstance(stage, int):
        raise RuntimeError('stage must be an int but received {}.'.format(
            type(stage).__name__))

    if stage < 0 or stage > 2:
        raise RuntimeError(
            'stage must be between 0 and 3 but received {}.'.format(stage))


def check_z_bin(z_bin):
    if not isinstance(z_bin, int):
        raise RuntimeError('z_bin must be an int but received {}.'.format(
            type(z_bin).__name__))

    if z_bin < 0 or z_bin > 3:
        raise RuntimeError(
            'z_bin must be between 0 and 3 but received {}.'.format(z_bin))


def source_z_bins(stage, survey=None):

    if stage == 0:
        return [0.5, 0.7, 0.9, 1.1, 1.5]

    if survey == 'des':
        return [0.20, 0.43, 0.63, 0.90, 1.30]
    elif survey == 'hsc':
        return [0.3, 0.6, 0.9, 1.2, 1.5]
    elif survey == 'kids':
        return [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    else:
        raise RuntimeError('Unkown survey: {}.'.format(survey))


def read_raw_data(stage, catalog_type, z_bin, survey=None):

    if catalog_type not in ['source', 'lens', 'calibration', 'random']:
        raise RuntimeError('Unkown catalog type: {}.'.format(catalog_type))

    if host == 'lux':
        path = os.path.join(os.sep, 'data', 'groups', 'leauthaud', 'jolange',
                            'Zebu', 'raw', 'stage{}mocks'.format(stage))
    elif host == 'cori':
        path = os.path.join(os.sep, 'project', 'projectdirs', 'desi', 'users',
                            'cblake', 'lensing', 'stage{}mocks'.format(stage))
    else:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'raw',
                            'stage{}mocks'.format(stage))

    z_bin_min = 0
    z_bin_max = 3

    if survey == 'kids' and catalog_type in ['source', 'calibration']:
        z_bin_max = 4

    if z_bin < z_bin_min or z_bin > z_bin_max:
        raise RuntimeError('Invalid {} redshift bin. '.format(catalog_type) +
                           'Must be in [{}, {}]'.format(z_bin_min, z_bin_max) +
                           ', but received {}.'.format(z_bin))

    if (stage == 0 or stage == 1) and catalog_type in ['lens', 'random']:
        z_bins = [0.1, 0.3, 0.5, 0.7, 0.9]

        cols_l = ['ra', 'dec', 'z', 'w_sys']

        z_min = '{:.1f}'.format(z_bins[z_bin]).replace('.', 'pt')
        z_max = '{:.1f}'.format(z_bins[z_bin + 1]).replace('.', 'pt')

        fname = 'stage{}mock_reg{}_{}_{}_zs{}_{}.dat'.format(
            stage, 1,  'lenses' if catalog_type == 'lens'
            else 'randlenses', 'BGS' if z_bin <= 1 else 'LRG', z_min, z_max)

        table = Table.read(os.path.join(path, fname), format='ascii',
                           data_start=1, names=cols_l if
                           catalog_type == 'source' else cols_l)

    elif stage == 0 and catalog_type in ['source', 'calibration']:

        z_bins = source_z_bins(0)

        cols_s = ['ra', 'dec', 'z_true', 'z', 'gamma_1', 'gamma_2', 'e_1',
                  'e_2', 'w']
        cols_c = ['z_true', 'z']

        z_min = '{:.1f}'.format(z_bins[z_bin]).replace('.', 'pt')
        z_max = '{:.1f}'.format(z_bins[z_bin + 1]).replace('.', 'pt')

        fname = 'stage0mock_{}_sources_zp{}_{}.dat'.format(
            'reg1' if catalog_type == 'source' else 'cal', z_min, z_max)

        table = Table.read(os.path.join(path, fname), format='ascii',
                           data_start=1, names=cols_s if
                           catalog_type == 'source' else cols_c)

        table['z_err'] = 0.1 * (1 + table['z_true'])

        if catalog_type == 'calibration':
            table['w'] = np.ones(len(table))
            table['w_sys'] = np.ones(len(table))

    elif stage == 1 and catalog_type in ['source', 'calibration']:

        cols_s = ['ra', 'dec', 'z_true', 'z', 'gamma_1', 'gamma_2', 'e_1',
                  'e_2', 'w']
        cols_c = ['z_true', 'z', 'w']

        if survey not in ['des', 'hsc', 'kids']:
            raise RuntimeError('Unkown survey: {}.'.format(survey))

        z_bins = source_z_bins(1, survey)

        if survey == 'des':
            cols_s += ['R_11', 'R_22']
            cols_c.remove('w')
            cols_c += ['R_11', 'R_22']
        elif survey == 'hsc':
            cols_s += ['m', 'sigma_rms']
            cols_c += ['m', 'sigma_rms']
        elif survey == 'kids':
            cols_s += ['m', 'dummy']
            cols_c += ['m']

        z_min = '{:.2f}'.format(z_bins[z_bin]).replace('.', 'pt')
        z_max = '{:.2f}'.format(z_bins[z_bin + 1]).replace('.', 'pt')

        fname = 'stage1mock_{}_sources_{}_zp{}_{}.dat'.format(
            'reg1' if catalog_type == 'source' else 'cal', survey, z_min,
            z_max)

        table = Table.read(os.path.join(path, fname), format='ascii',
                           data_start=1, names=cols_s if
                           catalog_type == 'source' else cols_c)

        if survey == 'des':
            table['R_MCAL'] = (table['R_11'] + table['R_22']) / 2

        if catalog_type == 'calibration':
            table['w_sys'] = np.ones(len(table))

        if survey == 'des' and catalog_type == 'calibration':
            table['w'] = np.ones(len(table))

    return table
