import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
rp_bins = 0.05 * np.logspace(0, 3, 31)


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


def raw_data_path(stage, catalog_type, z_bin, survey=None):

    if catalog_type not in ['source', 'lens', 'calibration', 'random']:
        raise RuntimeError('Unkown catalog type: {}.'.format(catalog_type))

    path = os.path.join(os.sep, 'project', 'projectdirs', 'desi', 'users',
                        'cblake', 'lensing', 'stage{}mocks'.format(stage))

    if stage == 0:
        z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]

        if catalog_type in ['lens', 'random']:
            z_min = '{:.1f}'.format(z_bins[z_bin]).replace('.', 'pt')
            z_max = '{:.1f}'.format(z_bins[z_bin + 1]).replace('.', 'pt')
            fname = 'stage0mock_reg1_{}_{}_zs{}_{}.dat'.format(
                'lenses' if catalog_type == 'lens' else 'randlenses',
                'BGS' if z_bin <= 1 else 'LRG', z_min, z_max)
        else:
            z_min = '{:.1f}'.format(z_bins[z_bin + 2]).replace('.', 'pt')
            z_max = '{:.1f}'.format(z_bins[z_bin + 3]).replace('.', 'pt')
            fname = 'stage0mock_cal_sources_zp{}_{}.dat'.format(z_min, z_max)

    return os.path.join(path, fname)
