import os
import camb
import numpy as np
from astropy import units as u
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COSMOLOGY = FlatLambdaCDM(Om0=0.286, H0=100)
RP_BINS = 0.2 * np.logspace(0, 2, 21)
THETA_BINS = 3 * np.logspace(0, 2, 21) * u.arcmin

SOURCE_Z_BINS = {
    'des': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}
LENS_Z_BINS = {
    'bgs': np.array([0.1, 0.3, 0.5]),
    'lrg': np.array([0.5, 0.7, 0.9])}

ALPHA_L = [0.818, 1.658, 2.568, 1.922]


def stacking_kwargs(survey, statistic='ds'):

    if survey.lower() in ['des', 'hsc', 'kids']:
        kwargs = {'boost_correction': False, 'random_subtraction': False,
                  'scalar_shear_response_correction': survey.lower() != 'des',
                  'matrix_shear_response_correction': survey.lower() == 'des',
                  'shear_responsivity_correction': survey.lower() == 'hsc'}
        if statistic == 'ds':
            kwargs['photo_z_dilution_correction'] = True
        elif statistic == 'gt':
            pass
        else:
            raise ValueError("Unknown statistic '{}'.".format(statistic))
        return kwargs
    else:
        raise RuntimeError('Unkown lensing survey {}.'.format(survey))


def read_mock_catalog(survey, magnification=True, fiber_assignment=False,
                      intrinsic_alignment=False, photometric_redshifts=True,
                      shear_bias=True, shape_noise=False):

    if shape_noise:
        if not (shear_bias and intrinsic_alignment):
            raise ValueError('If `shape_noise` is true, `shear_bias` and ' +
                             '`intrinsic_alignment` must also be true.')

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
            if survey not in ['des-c', 'hsc-c', 'kids-c']:
                path = survey
            else:
                path = survey.split('-')[0]
            cat_all[survey].append(Table.read(
                os.path.join(fpath, fname), path=path))

    for survey in survey_list:
        if survey in ['bgs-r', 'lrg-r']:
            continue
        for i in range(len(cat_all['buzzard'])):
            cat_survey = cat_all[survey][i]
            cat_buzzard = cat_all['buzzard'][i][cat_survey['id_buzzard']]
            for key in ['ra', 'dec', 'mu', 'g_1', 'g_2', 'ia_1', 'ia_2']:
                cat_survey[key] = cat_buzzard[key]
            cat_survey['z_true'] = cat_buzzard['z']

    for survey in survey_list:
        meta = cat_all[survey][0].meta
        if 'area' in cat_all[survey][0].meta.keys():
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
        if survey not in ['des', 'hsc', 'kids', 'des-c', 'hsc-c',
                          'kids-c']:
            continue

        if not shape_noise or survey in ['des-c', 'hsc-c', 'kids-c']:
            cat_survey = cat_all[survey]

            if not shear_bias:
                if 'm' in cat_all[survey].colnames:
                    cat_survey['m'] = 0
                if 'e_rms' in cat_all[survey].colnames:
                    cat_survey['e_rms'] = 0
                if 'R_11' in cat_all[survey].colnames:
                    cat_survey['R_11'] = 1
                if 'R_22' in cat_all[survey].colnames:
                    cat_survey['R_22'] = 1
                if 'R_21' in cat_all[survey].colnames:
                    cat_survey['R_21'] = 0
                if 'R_12' in cat_all[survey].colnames:
                    cat_survey['R_12'] = 0

            r_1 = np.ones(len(cat_survey))
            r_2 = np.ones(len(cat_survey))
            if 'm' in cat_survey.colnames:
                r_1 *= 1 + cat_survey['m']
                r_2 *= 1 + cat_survey['m']
            if 'R_11' in cat_survey.colnames:
                r_1 *= 0.5 * (cat_survey['R_11'] + cat_survey['R_22'])
                r_2 *= 0.5 * (cat_survey['R_11'] + cat_survey['R_22'])
            if 'e_rms' in cat_survey.colnames:
                r_1 *= 2 * (1 - cat_survey['e_rms']**2)
                r_2 *= 2 * (1 - cat_survey['e_rms']**2)

            if survey in ['des-c', 'hsc-c', 'kids-c']:
                cat_survey['w_sys'] = (r_1 + r_2) / 2.0

            if not shape_noise:
                cat_survey['e_1'] = cat_survey['g_1'] * r_1
                cat_survey['e_2'] = cat_survey['g_2'] * r_2
                if intrinsic_alignment:
                    cat_survey['e_1'] += cat_survey['ia_1'] * r_1
                    cat_survey['e_2'] += cat_survey['ia_2'] * r_2

        cat_survey['e_2'] = - cat_survey['e_2']

    for survey in survey_list:
        if survey in ['bgs', 'lrg']:
            cat_all[survey].rename_column('z_true', 'z')
        if survey in ['des-c', 'hsc-c', 'kids-c']:
            cat_all[survey] = cat_all[survey][::100]

    for survey in survey_list:
        columns_keep = ['ra', 'dec', 'e_1', 'e_2', 'z_true', 'z', 'e_rms',
                        'm', 'R_11', 'R_22', 'R_12', 'R_21', 'w', 'w_sys']
        for key in cat_all[survey].colnames:
            if key not in columns_keep:
                cat_all[survey].remove_column(key)
            else:
                cat_all[survey][key] = cat_all[survey][key].astype(float)

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
