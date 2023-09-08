import camb
import numpy as np

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from mocks.read_mocks import read_mock_catalog
from pathlib import Path


BASE_PATH = Path(__file__).absolute().parent
MOCK_PATH = BASE_PATH / 'mocks'
PIXELS = np.genfromtxt(MOCK_PATH / 'pixels.csv', dtype=int)
COSMOLOGY = FlatLambdaCDM(Om0=0.286, H0=100)
RP_BINS = np.geomspace(0.08, 80, 16)
THETA_BINS = np.geomspace(3, 300, 16) * u.arcmin
ABS_MAG_R_MAX = [-19.5, -20.5, -21]

SOURCE_Z_BINS = {
    'des': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}
LENS_Z_BINS = {
    'bgs': np.array([0.1, 0.2, 0.3, 0.4]),
    'lrg': np.array([0.4, 0.6, 0.8])}

ALPHA_L = [0.7888095257585374, 0.9502395911520835, 1.4299332686091044,
           2.58415519575035, 2.257679328458963]


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

    h = 0.7
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                       omch2=(COSMOLOGY.Om0 - 0.046) * h**2)
    pars.InitPower.set_params(ns=0.96, As=2.179661598125922e-09)
    pars.set_matter_power(redshifts=np.linspace(2, 0, 20), kmax=2000.0,
                          nonlinear=True)

    return camb.get_results(pars)


def errors():

    err = dict()

    for statistic in ['ds', 'gt']:

        err[statistic] = dict()

        for sources in ['des', 'hsc', 'kids']:

            err[statistic][sources] = dict()

            for lenses in ['bgs', 'lrg']:

                err[statistic][sources][lenses] = np.zeros((
                    len(SOURCE_Z_BINS[sources]) - 1,
                    len(LENS_Z_BINS[lenses]) - 1,
                    len(RP_BINS if statistic == 'ds' else THETA_BINS) - 1))

                if sources == 'des':
                    sources_file = 'desy3'
                elif sources == 'kids':
                    sources_file = 'kids1000'
                else:
                    sources_file = 'hsc'

                table_bin = Table.read(
                    MOCK_PATH / 'theory' / 'bin_{}_{}desiy1{}.dat'.format(
                        statistic, sources_file, lenses), format='ascii',
                    delimiter=' ')
                for i, name in enumerate(['bin', 'lens_bin', 'source_bin',
                                          'radial_bin', 'r']):
                    table_bin.rename_column('col{}'.format(i + 1), name)

                table_err = Table.read(
                    MOCK_PATH / 'theory' /
                    '{}erranavec_{}desiy1{}{}.dat'.format(
                        statistic, sources_file, lenses, '_pzwei' if
                        statistic == 'ds' else ''), format='ascii',
                    delimiter=' ')
                table_err.rename_column('col2', 'error')

                for i, (l, s, r) in enumerate(zip(
                        table_bin['lens_bin'], table_bin['source_bin'],
                        table_bin['radial_bin'])):
                    if statistic == 'ds' and r >= len(RP_BINS):
                        continue
                    if l > 2 and lenses == 'lrg':
                        continue
                    err[statistic][sources][lenses][s - 1, l - 1, r - 1] =\
                        table_err['error'][i]

    return err
