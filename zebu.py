import camb
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

from mocks.read_mocks import read_mock_catalog


BASE_PATH = Path(__file__).absolute().parent
MOCK_PATH = BASE_PATH / 'mocks'
PIXELS = np.genfromtxt(MOCK_PATH / 'pixels.csv', dtype=int)
COSMOLOGY = FlatLambdaCDM(Om0=0.286, H0=100)
RP_BINS = np.logspace(0, 2, 21)[:-7]
THETA_BINS = 3 * np.logspace(0, 2, 21)[:-8] * u.arcmin

SOURCE_Z_BINS = {
    'des': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}
LENS_Z_BINS = {
    'bgs': np.array([0.1, 0.3, 0.5]),
    'lrg': np.array([0.5, 0.7, 0.9])}

ALPHA_L = [0.7642979832435776, 1.6389627681213133, 2.497741720186678,
           1.953780096692544]


def stacking_kwargs(survey, statistic='ds'):

    if survey.lower() in ['des', 'hsc', 'kids']:
        kwargs = {'boost_correction': False, 'random_subtraction': True,
                  'scalar_shear_response_correction': True,
                  'matrix_shear_response_correction': survey.lower() == 'des',
                  'shear_responsivity_correction': survey.lower() == 'hsc'}
        if statistic == 'ds':
            if survey == 'hsc':
                kwargs['photo_z_dilution_correction'] = True
        elif statistic == 'gt':
            pass
        else:
            raise ValueError("Unknown statistic '{}'.".format(statistic))
        return kwargs
    else:
        raise RuntimeError('Unkown lensing survey {}.'.format(survey))


def get_camb_results():

    h = COSMOLOGY.H0.value / 100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                       omch2=(COSMOLOGY.Om0 - 0.046) * h**2)
    pars.InitPower.set_params(ns=0.96, As=9.93075e-10)
    pars.set_matter_power(redshifts=np.linspace(2, 0, 20), kmax=2000.0,
                          nonlinear=True)

    return camb.get_results(pars)
