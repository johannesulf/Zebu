import os
import camb
import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COSMOLOGY = FlatLambdaCDM(Om0=0.286, H0=100)
RP_BINS = 0.2 * np.logspace(0, 2, 21)
THETA_BINS = 3 * np.logspace(0, 2, 21)

LENS_Z_BINS = np.linspace(0.1, 0.9, 5)
SOURCE_Z_BINS = {
    'des': np.array([0.2, 0.43, 0.63, 0.9, 1.3]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}

ALPHA_L = [0.818, 1.658, 2.568, 1.922]


def stacking_kwargs(survey):

    if survey.lower() in ['des', 'hsc', 'kids']:
        return {'photo_z_dilution_correction': True,
                'boost_correction': False, 'random_subtraction': False,
                'scalar_shear_response_correction': survey.lower() != 'des',
                'matrix_shear_response_correction': survey.lower() == 'des',
                'shear_responsivity_correction': survey.lower() == 'hsc'}
    else:
        raise RuntimeError('Unkown lensing survey {}.'.format(survey))


def read_mock_catalog(survey, magnification=True, fiber_assignment=False,
                      intrinsic_alignment=False, photometric_redshift=True,
                      shear_bias=True, shape_noise=False):

    if isinstance(survey, str):
        survey_list = [survey, ]
    else:
        survey_list = survey

    if isinstance(magnification, bool):
        magnification_list = [magnification, ] * len(survey_list)
    else:
        magnification_list = magnification

    fpath = os.path.join(BASE_DIR, 'mocks', 'mocks')
    fname_list = os.listdir(fpath)

    cat_all = {}
    for survey in ['buzzard', ] + survey_list:
        cat_all[survey] = []
        for fname in fname_list:
            if fname[:5] != 'pixel':
                continue
            cat_all[survey].append(Table.read(
                os.path.join(fpath, fname), path=survey))

    for survey in survey_list:
        if survey in ['bgs-r', 'lrg-r']:
            continue
        for i in range(len(cat_all['buzzard'])):
            cat_survey = cat_all[survey][i]
            cat_buzzard = cat_all['buzzard'][i][cat_survey['id_buzzard']]
            for key in ['ra', 'dec', 'mu', 'g_1', 'g_2']:
                if shear_bias and key in ['g_1', 'g_2']:
                    continue
                cat_survey[key] = cat_buzzard[key]
            cat_survey['z_true'] = cat_buzzard['z']

    for survey in survey_list:
        meta = cat_all[survey][0].meta
        meta['area'] = np.sum([c.meta['area'] for c in cat_all[survey]])
        cat_all[survey] = vstack(cat_all[survey], metadata_conflicts='silent')
        cat_all[survey].meta = meta

    for survey in survey_list:
        if survey in ['bgs', 'lrg', 'bgs-r', 'lrg-r']:
            cat_all[survey]['w_sys'] = np.ones(len(cat_all[survey]))

    for survey, magnification in zip(survey_list, magnification_list):
        if survey in ['bgs-r', 'lrg-r']:
            continue
        if magnification:
            cat_all[survey] = cat_all[survey][cat_all[survey]['target']]
        else:
            cat_all[survey] = cat_all[survey][cat_all[survey]['target_t']]
            if survey in ['bgs', 'lrg']:
                key = 'w_sys'
            else:
                key = 'w'
            cat_all[survey][key] = cat_all[survey][key] * cat_all[survey]['mu']

    for survey in survey_list:
        if survey in ['bgs', 'lrg']:
            cat_all[survey].rename_column('z_true', 'z')
        if survey in ['des-c', 'hsc-c', 'kids-c']:
            cat_all[survey] = cat_all[survey][::100]
        elif survey in ['des', 'hsc', 'kids', 'des-c', 'hsc-c', 'kids-c']:
            if not photometric_redshift:
                cat_all[survey]['z'] = cat_all[survey]['z_true']
            if not shear_bias:
                if 'm' in cat_all[survey].colnames:
                    cat_all[survey]['m'] = 0
                if 'e_rms' in cat_all[survey].colnames:
                    cat_all[survey]['e_rms'] = 0
                if 'R_11' in cat_all[survey].colnames:
                    cat_all[survey]['R_11'] = 1
                if 'R_22' in cat_all[survey].colnames:
                    cat_all[survey]['R_22'] = 1
                if 'R_21' in cat_all[survey].colnames:
                    cat_all[survey]['R_21'] = 0
                if 'R_12' in cat_all[survey].colnames:
                    cat_all[survey]['R_12'] = 0
            if not shape_noise:
                cat_all[survey]['e_1'] = cat_all[survey]['g_1']
                cat_all[survey]['e_2'] = cat_all[survey]['g_2']

    map_ia = None
    red_ia = None

    for survey in survey_list:
        if survey not in ['des', 'hsc', 'kids'] or not intrinsic_alignment:
            continue
        cat_survey = cat_all[survey]

        r_1 = np.ones(len(cat_survey))
        r_2 = np.ones(len(cat_survey))
        if survey in ['hsc', 'kids']:
            r_1 *= 1 + cat_survey['m']
            r_2 *= 1 + cat_survey['m']
        if survey == 'des':
            r_1 *= 0.5 * (cat_survey['R_11'] + cat_survey['R_22'])
            r_2 *= 0.5 * (cat_survey['R_11'] + cat_survey['R_22'])
        if survey == 'hsc':
            r_1 *= 2 * (1 - cat_survey['e_rms']**2)
            r_2 *= 2 * (1 - cat_survey['e_rms']**2)

        if map_ia is None:
            map_ia = Table.read(os.path.join(fpath, 'ia.hdf5'), path='ia')
            red_ia = Table.read(os.path.join(fpath, 'ia.hdf5'), path='z')

        pix = hp.ang2pix(2048, cat_survey['ra'], cat_survey['dec'],
                         lonlat=True)
        row = np.searchsorted(map_ia['pix'], pix)

        for i in range(len(red_ia)):
            select = ((cat_survey['z_true'] >= red_ia['z_min'][i]) &
                      (cat_survey['z_true'] < red_ia['z_max'][i]))

            cat_survey['e_1'] += map_ia['e_1'][:, i][row] * r_1 * select
            cat_survey['e_2'] += map_ia['e_2'][:, i][row] * r_2 * select

    for survey in survey_list:
        columns_keep = ['ra', 'dec', 'e_1', 'e_2', 'z_true', 'z', 'e_rms',
                        'm', 'R_11', 'R_22', 'R_12', 'R_21', 'w', 'w_sys']
        for key in cat_all[survey].colnames:
            if key not in columns_keep:
                cat_all[survey].remove_column(key)

    cat_all = [cat_all[survey] for survey in survey_list]
    if len(cat_all) == 1:
        cat_all = cat_all[0]

    return cat_all


def get_camb_results():

    h = COSMOLOGY.H0.value / 100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                       omch2=(COSMOLOGY.Om0 - 0.046) * h**2)
    pars.InitPower.set_params(ns=0.96, As=9.93075e-10)
    pars.set_matter_power(redshifts=np.linspace(2, 0, 20), kmax=2000.0,
                          nonlinear=True)

    return camb.get_results(pars)
