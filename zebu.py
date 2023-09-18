import camb
import numpy as np

from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack
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

ALPHA_L = [0.9145633761955659, 1.5806488673171675, 2.0206666166515483,
           2.58339234742153, 2.259265182806545]


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


def covariance(statistic, sources):

    if sources == 'des':
        sources = 'desy3'
    elif sources == 'kids':
        sources = 'kids1000'
    else:
        sources = 'hscy1'

    table_bin_1 = Table.read(
        MOCK_PATH / 'theory' / 'bin_{}_{}desiy1{}.dat'.format(
            statistic, sources, 'bgs'), format='ascii', delimiter=' ')
    table_bin_2 = Table.read(
        MOCK_PATH / 'theory' / 'bin_{}_{}desiy1{}.dat'.format(
            statistic, sources, 'lrg'), format='ascii', delimiter=' ')
    table_bin_2['col1'] += len(table_bin_1)
    table_bin_2['col2'] += 3
    table_bin = vstack([table_bin_1, table_bin_2])
    for i, name in enumerate(['bin', 'lens_bin', 'source_bin',
                              'radial_bin', 'r']):
        table_bin.rename_column('col{}'.format(i + 1), name)
        if i != 4:
            table_bin[name] -= 1

    cov = np.zeros((len(table_bin), len(table_bin)))

    cov_bgs = np.genfromtxt(
        MOCK_PATH / 'theory' / '{}covcorr_{}desiy1{}{}.dat'.format(
            statistic, sources, 'bgs', '_pzwei' if statistic == 'ds' else ''),
        skip_header=1)[:, -1]
    cov_bgs = cov_bgs.reshape(int(np.sqrt(len(cov_bgs))),
                              int(np.sqrt(len(cov_bgs))))
    cov[:len(cov_bgs), :len(cov_bgs)] = cov_bgs

    cov_lrg = np.genfromtxt(
        MOCK_PATH / 'theory' / '{}covcorr_{}desiy1{}{}.dat'.format(
            statistic, sources, 'lrg', '_pzwei' if statistic == 'ds' else ''),
        skip_header=1)[:, -1]
    cov_lrg = cov_lrg.reshape(int(np.sqrt(len(cov_lrg))),
                              int(np.sqrt(len(cov_lrg))))
    cov[-len(cov_lrg):, -len(cov_lrg):] = cov_lrg

    return cov, table_bin
