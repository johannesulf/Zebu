import camb
import numpy as np
from pathlib import Path
from astropy import units as u
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM


BASE_PATH = Path(__file__).absolute().parent
MOCK_PATH = BASE_PATH / 'mocks' / 'mocks'
MOCK_PIXEL_LIST = []
for fname in MOCK_PATH.iterdir():
    if fname.suffix == '.hdf5':
        MOCK_PIXEL_LIST.append(
            int(str(fname.stem).split('.')[-1].split('_')[1]))
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
                  'scalar_shear_response_correction': True,
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
                      intrinsic_alignment=False, shear_bias=True,
                      shape_noise=False, path=MOCK_PATH / 'buzzard-4',
                      pixel=MOCK_PIXEL_LIST):
    """
    Read in one or multiple mock catalogs.

    Attributes
    ----------
    survey : string or list
        Mock survey to read. Options are 'bgs', 'bgs-r' (BGS randoms), 'lrg',
        'lrg-r', 'des', 'des-c' (DES calibration sample), 'hsc', 'hsc-c',
        'kids' and 'kids-c'. You can specify multiple surveys at once by
        passing a list.
    magnification : bool or list, optional
        Whether to include magnification effects. If list, specifies whether
        magnification effects are included for each survey. Default is True.
    fiber_assignment : bool, optional
        Whether to include fiber assignment for BGS and LRG. NOT IMPLEMENTED,
        YET. Default is False.
    intrinsic_alignment : bool, optional
        Whether to include intrinsic alignment effects on the measured shears.
        Default is False.
    shear_bias : bool, optional
        Whether to include shear bias effects on the measured ellipticities.
        Default is True.
    shape_noise : bool, optional
        Whether to include shape noise on the measured ellipticities. Default
        is False.
    path : pathlib.Path, optional
        Path to the folder in which the mocks are located in.
    pixel : list
        List of Healpix pixels to read

    Returns
    -------
    mocks : astropy.Table or list
        One or more mock survey catalogs, depending on whether one or more
        surveys were requested. The tables have the following columns.
        * 'ra': (lensed) right ascension
        * 'dec': (lensed) declination
        * 'z': measured redshift
        * 'z_true': true cosmological redshift
        * 'w_sys': lens or calibration systematic weight
        * 'w': source weight
        * 'e_1'/'e_2': ellipticity components
        * 'g_1'/'g_2': true shear components
        * 'ia_1'/'ia_2': IA components
        * 'e_rms'/'m'/'R_11'/'R_22'/'R_21'/'R_12': shear biases

    """
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

    table_all = {}
    for survey in ['buzzard', ] + survey_list:
        table_all[survey] = []
        for p in pixel:
            # Remove '-c' for calibration samples, e.g, des-c.
            table_all[survey].append(Table.read(
                path / 'pixel_{}.hdf5'.format(p), path=survey.split('-c')[0]))

    for survey in survey_list:
        if survey in ['bgs-r', 'lrg-r']:
            continue
        for i in range(len(table_all['buzzard'])):
            table_survey = table_all[survey][i]
            table_buzzard = table_all['buzzard'][i][table_survey['id_buzzard']]
            for key in ['ra', 'dec', 'mu', 'g_1', 'g_2', 'ia_1', 'ia_2']:
                table_survey[key] = table_buzzard[key]
            table_survey['z_true'] = table_buzzard['z']

    for survey in survey_list:
        meta = table_all[survey][0].meta
        if 'area' in table_all[survey][0].meta.keys():
            meta['area'] = np.sum([c.meta['area'] for c in table_all[survey]])
        table_all[survey] = vstack(
            table_all[survey], metadata_conflicts='silent')
        table_all[survey].meta = meta

    for survey in survey_list:
        if survey in ['bgs', 'lrg', 'bgs-r', 'lrg-r']:
            table_all[survey]['w_sys'] = np.ones(len(table_all[survey]))

    for survey, magnification in zip(survey_list, magnification_list):
        if survey in ['bgs-r', 'lrg-r']:
            continue
        if magnification:
            table_all[survey] = table_all[survey][table_all[survey]['target']]
        else:
            table_all[survey] = table_all[survey][
                table_all[survey]['target_t']]
            if survey in ['bgs', 'lrg']:
                key = 'w_sys'
            else:
                key = 'w'
            table_all[survey][key] = table_all[survey][key] * \
                table_all[survey]['mu']

    for survey in survey_list:
        if survey not in ['des', 'hsc', 'kids', 'des-c', 'hsc-c',
                          'kids-c']:
            continue

        if not shape_noise or survey in ['des-c', 'hsc-c', 'kids-c']:
            table_survey = table_all[survey]

            if not shear_bias:
                if 'm' in table_all[survey].colnames:
                    table_survey['m'] = 0
                if 'e_rms' in table_all[survey].colnames:
                    table_survey['e_rms'] = np.sqrt(0.5)
                if 'R_11' in table_all[survey].colnames:
                    table_survey['R_11'] = 1
                if 'R_22' in table_all[survey].colnames:
                    table_survey['R_22'] = 1
                if 'R_21' in table_all[survey].colnames:
                    table_survey['R_21'] = 0
                if 'R_12' in table_all[survey].colnames:
                    table_survey['R_12'] = 0

            r = np.ones(len(table_survey))
            if 'm' in table_survey.colnames:
                r *= 1 + table_survey['m']
            if 'R_11' in table_survey.colnames:
                r *= 0.5 * (table_survey['R_11'] + table_survey['R_22'])
            if 'e_rms' in table_survey.colnames:
                r *= 2 * (1 - table_survey['e_rms']**2)

            if survey in ['des-c', 'hsc-c', 'kids-c']:
                table_survey['w_sys'] = r

            if not shape_noise:
                table_survey['e_1'] = table_survey['g_1'] * r
                table_survey['e_2'] = table_survey['g_2'] * r
                if intrinsic_alignment:
                    table_survey['e_1'] += table_survey['ia_1'] * r
                    table_survey['e_2'] += table_survey['ia_2'] * r

        table_survey['e_2'] = - table_survey['e_2']

    for survey in survey_list:
        if survey in ['bgs', 'lrg']:
            table_all[survey].rename_column('z_true', 'z')
        if survey in ['des-c', 'hsc-c', 'kids-c']:
            table_all[survey] = table_all[survey][::100]

    for survey in survey_list:
        columns_keep = ['ra', 'dec', 'e_1', 'e_2', 'z_true', 'z', 'e_rms',
                        'm', 'R_11', 'R_22', 'R_12', 'R_21', 'w', 'w_sys',
                        'g_1', 'g_2', 'ia_1', 'ia_2']
        for key in table_all[survey].colnames:
            if key not in columns_keep:
                table_all[survey].remove_column(key)
            else:
                table_all[survey][key] = table_all[survey][key].astype(float)

    table_all = [table_all[survey] for survey in survey_list]
    if len(table_all) == 1:
        table_all = table_all[0]

    return table_all


def get_camb_results():

    h = COSMOLOGY.H0.value / 100
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=0.046 * h**2,
                       omch2=(COSMOLOGY.Om0 - 0.046) * h**2)
    pars.InitPower.set_params(ns=0.96, As=9.93075e-10)
    pars.set_matter_power(redshifts=np.linspace(2, 0, 20), kmax=2000.0,
                          nonlinear=True)

    return camb.get_results(pars)
