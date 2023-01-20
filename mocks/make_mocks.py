import os
import tqdm
import fitsio
import argparse
import numpy as np
import healpy as hp
from pathlib import Path
from astropy.table import Table
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.interpolate import splev, splrep
from astropy.cosmology import FlatLambdaCDM


SOURCE_Z_BINS = {
    'des': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}
MAG_BINS = np.linspace(18, 26, 81)
TABLE_S = {}
TABLE_C = {}
TABLE_R = {}
TABLE_IA = None
BUZZARD_PATH = None
MOCK_INPUT_PATH = (Path(os.getenv('CFS')) / 'desi' / 'users' / 'cblake' /
                   'lensing' / 'mock_inputs')


def read_buzzard_catalog(pixel):

    path = BUZZARD_PATH / 'truth'
    fname = 'Chinchilla-4_cam_rs_scat_shift_lensed.{}.fits'.format(pixel)

    columns = ['GAMMA1', 'GAMMA2', 'Z', 'MU', 'RA', 'DEC', 'SIZE', 'TSIZE']
    table = Table(fitsio.read(path / fname, columns=columns))
    table.rename_column('GAMMA1', 'g_1')
    table.rename_column('GAMMA2', 'g_2')
    table.rename_column('Z', 'z')
    table.rename_column('MU', 'mu')
    table.rename_column('RA', 'ra')
    table.rename_column('DEC', 'dec')
    table.rename_column('SIZE', 'size')
    table.rename_column('TSIZE', 'size_t')

    path = BUZZARD_PATH / 'surveymags'
    fname = 'Chinchilla-4-aux.{}.fits'.format(pixel)

    mag = fitsio.read(path / fname, columns=['LMAG'])['LMAG']
    mag_t = fitsio.read(path / fname, columns=['TMAG'])['TMAG']

    table['mag'] = mag[:, [1, 2, 3, 4, 5, -2, -1]]
    table['mag_t'] = mag_t[:, [1, 2, 3, 4, 5, -2, -1]]

    for suffix in ['', '_t']:
        size = table['size' + suffix]
        table['mag_fib_r' + suffix] = (
            table['mag' + suffix][:, 1] -
            2.5 * np.log10((1 - np.exp(-0.275 / size)) / (1 - np.exp(-1))))
        table['mag_fib_z' + suffix] = (
            table['mag' + suffix][:, 3] -
            2.5 * np.log10((1 - np.exp(-0.105 / size)) / (1 - np.exp(-1))))

    table.meta = {}

    for key in table.colnames:
        table[key] = table[key].astype(np.float32)

    table.meta['area'] = hp.nside2pixarea(8, degrees=True)
    table.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']

    return table


def add_ia_information(table_b):

    pixel = hp.ang2pix(2048, table_b['ra'], table_b['dec'],  lonlat=True)
    row = np.searchsorted(TABLE_IA['pix'], pixel)

    table_b['ia_1'] = np.zeros(len(table_b), dtype=np.float32)
    table_b['ia_2'] = np.zeros(len(table_b), dtype=np.float32)
    z_bins = TABLE_IA.meta['z_bins']

    for i in range(len(z_bins) - 1):
        select = (table_b['z'] >= z_bins[i]) & (table_b['z'] < z_bins[i + 1])
        table_b['ia_1'] += TABLE_IA['ia_1'][:, i][row] * select
        table_b['ia_2'] += TABLE_IA['ia_2'][:, i][row] * select

    return table_b


def read_real_source_catalog(survey):

    if survey.lower() not in ['des', 'kids', 'hsc']:
        raise RuntimeError('Unkown survey {}.'.format(survey))
    elif survey.lower() == 'des':
        survey = 'desy3'

    fname = '{}_mag.fits'.format(survey.lower())
    table_s = Table.read(MOCK_INPUT_PATH / fname)

    if survey == 'desy3':

        table_s.meta['area'] = 4143.
        table_s.meta['bands'] = 'griz'

        table_s['mag'] = np.zeros((len(table_s), len(table_s.meta['bands'])))
        for i, band in enumerate(table_s.meta['bands']):
            table_s['mag'][:, i] = table_s['mag_{}'.format(band)]
        table_s.rename_column('wei', 'w')
        table_s.rename_column('R11', 'R_11')
        table_s.rename_column('R22', 'R_22')
        table_s['R_12'] = np.zeros(len(table_s))
        table_s['R_21'] = np.zeros(len(table_s))
        z_bin = table_s['tombin']
        table_s['z'] = (
            SOURCE_Z_BINS['des'][z_bin] +
            np.random.random(len(table_s)) *
            np.diff(SOURCE_Z_BINS['des'])[z_bin])

        table_s.keep_columns(['mag', 'w', 'R_11', 'R_22', 'R_12', 'R_21', 'z'])

        use = (0 <= z_bin) & (z_bin < len(SOURCE_Z_BINS['des']))
        use = use & (table_s['R_11'] > -1000.) & (table_s['R_22'] > -1000.)
        table_s = table_s[use]

    elif survey == 'hsc':

        table_s.meta['area'] = 158.3
        table_s.meta['bands'] = 'grizy'

        table_s['mag'] = np.zeros((len(table_s), len(table_s.meta['bands'])))
        for i, band in enumerate(table_s.meta['bands']):
            table_s['mag'][:, i] = table_s['{}cmodel_mag'.format(band)]
        table_s.rename_column('weight', 'w')
        table_s.rename_column('mcorr', 'm')
        table_s.rename_column('erms', 'e_rms')
        table_s.rename_column('photoz_best', 'z')

        table_s.keep_columns(['mag', 'w', 'm', 'e_rms', 'z'])
        use = (0.3 <= table_s['z']) & (table_s['z'] < 1.5)
        table_s = table_s[use]

    elif survey == 'kids':

        table_s.meta['area'] = 374.7
        table_s.meta['bands'] = 'grizy'

        table_s['mag'] = np.zeros((len(table_s), len(table_s.meta['bands'])))
        for i, band in enumerate(table_s.meta['bands']):
            table_s['mag'][:, i] = table_s['MAG_GAAP_{}'.format(
                band if i < 3 else band.upper())]

        table_s.rename_column('weight', 'w')
        table_s.rename_column('Z_B', 'z')
        table_s['z'] += np.random.uniform(-0.005, 0.005, len(table_s))

        table_s.keep_columns(['mag', 'w', 'z'])
        use = (0.1 <= table_s['z']) & (table_s['z'] < 1.2)
        table_s = table_s[use]

    return table_s


def read_real_calibration_catalog(survey):

    if survey.lower() not in ['des', 'kids', 'hsc']:
        raise RuntimeError('Unkown survey {}.'.format(survey))
    elif survey.lower() == 'des':
        survey = 'desy3'

    fname = '{}_cal.fits'.format(survey.lower())
    table_c = Table.read(MOCK_INPUT_PATH / fname)

    if survey == 'desy3':

        table_c.rename_column('zphot', 'z')
        table_c.rename_column('zspec', 'z_true')
        table_c['w_sys'] = 1.0

    elif survey == 'hsc':

        table_c.rename_column('redhsc', 'z')
        table_c.rename_column('redcosmos', 'z_true')
        table_c['w_sys'] = table_c['weisom'] * table_c['weilens']

    elif survey == 'kids':

        table_c.rename_column('z_B', 'z')
        table_c.rename_column('z_spec', 'z_true')
        table_c.rename_column('spec_weight_CV', 'w_sys')
        table_c['z'] += np.random.uniform(-0.005, 0.005, len(table_c))

    table_c.keep_columns(['z', 'z_true', 'w_sys'])
    table_c = table_c[table_c['w_sys'] > 0]

    return table_c


def read_random_catalog(survey):

    path = BUZZARD_PATH / 'desi_targets'
    fname = '{}_rand.fits'.format(survey)

    table = Table(fitsio.read(path / fname, columns=['ra', 'dec', 'redshift']))
    table.rename_column('redshift', 'z')

    for key in table.colnames:
        table[key] = table[key].astype(np.float32)

    return table


def photometric_redshift(z, survey):

    table_c = TABLE_C[survey]

    idx_sort = np.argsort(z)
    idx_inv_sort = np.argsort(idx_sort)
    z = z[idx_sort]

    z_phot = np.repeat(np.nan, len(z))
    z_bins = np.linspace(0.0, 2.0, 41)
    z_hist = np.histogramdd(
        np.vstack([table_c['z_true'], table_c['z']]).T,
        bins=(z_bins, z_bins))[0]

    for i in range(len(z_bins) - 1):

        i_min = np.searchsorted(z, z_bins[i])
        i_max = np.searchsorted(z, z_bins[i+1])

        pdf = z_hist[i, :]
        if np.sum(pdf) == 0:
            continue
        z_pdf = np.linspace(0.0, 2.0, 10000)
        pdf = splev(z_pdf, splrep(0.5 * (z_bins[1:] + z_bins[:-1]), pdf))
        pdf = np.maximum(pdf, 0)
        cdf = np.cumsum(pdf) / np.sum(pdf)

        z_phot[i_min:i_max] = z_pdf[
            np.searchsorted(cdf, np.random.random(i_max - i_min))]

    return z_phot[idx_inv_sort]


def detection_probability(table_b, survey):

    table_s = TABLE_S[survey]

    bins = (MAG_BINS, SOURCE_Z_BINS[survey])

    if survey == 'des':
        band = 'r'
        magshift = np.array([0.3, 0.3, 0.3, 0.2])
    elif survey == 'hsc':
        band = 'i'
        magshift = np.array([0, 0, 0, 0.1])
    else:
        band = 'r'
        magshift = np.array([0, 0, 0, 0, 0])

    mag_b = table_b['mag'][:, table_b.meta['bands'].index(band)]
    z_b = table_b['z_' + survey]
    z_dig = np.digitize(z_b, SOURCE_Z_BINS[survey]) - 1
    z_dig = np.minimum(np.maximum(z_dig, 0), len(SOURCE_Z_BINS[survey]) - 2)
    mag_b -= magshift[z_dig]
    hist_b = np.histogramdd(np.vstack([mag_b, z_b]).T, bins=bins)[0]

    mag_s = table_s['mag'][:, table_s.meta['bands'].index(band)]
    z_s = table_s['z']
    hist_s = np.histogramdd(np.vstack([mag_s, z_s]).T, bins=bins)[0]

    return (hist_s / table_s.meta['area']) / (hist_b / table_b.meta['area'])


def source_target(table_b, table_p, survey, random, lensed=True):

    suffix = '' if lensed else '_t'

    if survey == 'hsc':
        band = 'i'
    else:
        band = 'r'

    mag = table_b['mag' + suffix][:, table_b.meta['bands'].index(band)]
    mag_dig = np.digitize(mag, MAG_BINS) - 1
    z_dig = np.digitize(table_b['z_' + survey], SOURCE_Z_BINS[survey]) - 1
    target = ((mag_dig >= 0) & (mag_dig <= len(MAG_BINS) - 2) & (z_dig >= 0) &
              (z_dig <= len(SOURCE_Z_BINS[survey]) - 2))

    target[target] = (table_p[survey][mag_dig[target], z_dig[target]] >
                      random[target])
    return target


def bgs_target(table, lensed=True):

    suffix = '' if lensed else '_t'
    g = table['mag' + suffix][:, table.meta['bands'].index('g')]
    r = table['mag' + suffix][:, table.meta['bands'].index('r')]
    z = table['mag' + suffix][:, table.meta['bands'].index('z')]
    w1 = table['mag' + suffix][:, table.meta['bands'].index('w1')]
    r_fib = table['mag_fib_r' + suffix]
    r_off = 0.05
    offset = 0.04

    bgs = g - r > -1
    bgs &= g - r < 4
    bgs &= r - z > -1
    bgs &= r - z < 4

    fmc = np.zeros(len(table), dtype=bool)
    fmc |= ((r_fib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((r_fib < 22.9) & (r < 20.0) & (r > 17.8))
    fmc |= ((r_fib < 2.9 + r) & (r > 20))
    bgs &= fmc

    bright = r < 19.5 + offset + r_off
    bright &= r > 12.0 - r_off
    bright &= r_fib > 15.0

    faint = np.ones(len(table), dtype=bool)
    faint &= r < 20.22
    faint &= r > 19.5 + offset + r_off

    # D. Schlegel - ChangHoon H. color selection to get a high redshift
    # success rate.
    schlegel_color = (z - w1) - 1.2 * (g - (r - offset)) + 1.2
    faint &= (r_fib < 20.75 + offset) | (
        (r_fib < 21.5 + offset) & (schlegel_color > 0.0))

    bgs &= (bright | faint)

    return bgs


def lrg_target(table, lensed=True):

    suffix = '' if lensed else '_t'
    g = table['mag' + suffix][:, table.meta['bands'].index('g')]
    r = table['mag' + suffix][:, table.meta['bands'].index('r')]
    z = table['mag' + suffix][:, table.meta['bands'].index('z')]
    w1 = table['mag' + suffix][:, table.meta['bands'].index('w1')]
    z_fib = table['mag_fib_z' + suffix]

    lrg = z - w1 > 0.8 * (r - z) - 0.6  # non-stellar cut
    lrg &= z_fib < 21.6   # faint limit
    lrg &= (g - w1 > 2.9) | (r - w1 > 1.8)  # low-z cuts
    lrg &= (((r - w1 > (w1 - 17.14) * 1.8 + 0.4) &
             (r - w1 > (w1 - 16.33) * 1.)) |
            (r - w1 > 3.3 + 0.275))  # double sliding cuts and high-z extension

    return lrg


def apply_observed_shear(table_s, survey=None):

    table_s_ref = TABLE_S[survey]
    n_max = 1000000
    if len(table_s_ref) > n_max:
        table_s_ref = table_s_ref[
            np.random.choice(len(table_s_ref), n_max, replace=False)]

    mag = table_s['mag'][:, [b in table_s_ref.meta['bands'] for b in
                             table_s.meta['bands']]]
    mag_ref = table_s_ref['mag']

    tree = cKDTree(mag_ref)
    idx = tree.query(mag, k=3)[1]
    idx = np.array(idx)
    idx = idx[np.arange(len(idx)), np.random.randint(3, size=len(idx))]

    for key in table_s_ref.colnames:
        if key in ['m', 'w', 'R_11', 'R_22', 'R_12', 'R_21',
                   'e_rms']:
            table_s[key] = table_s_ref[key][idx]

    if survey.lower() in ['des', 'kids']:
        if survey.lower() == 'des':
            m = np.array([-0.006, -0.020, -0.024, -0.037])
        else:
            m = np.array([-0.009, -0.011, -0.015, 0.002, 0.007])
        z_dig = np.digitize(
            table_s['z'], SOURCE_Z_BINS[survey.lower()]) - 1
        table_s['m'] = m[z_dig]

    if survey.lower() in ['hsc', 'kids']:
        table_s['g_1'] *= 1 + table_s['m']
        table_s['g_2'] *= 1 + table_s['m']

    if survey.lower() == 'des':
        table_s['g_1'] *= 0.5 * (table_s['R_11'] + table_s['R_22'])
        table_s['g_2'] *= 0.5 * (table_s['R_11'] + table_s['R_22'])

    if survey.lower() == 'hsc':
        table_s['g_1'] *= 2 * (1 - table_s['e_rms']**2)
        table_s['g_2'] *= 2 * (1 - table_s['e_rms']**2)

    return table_s


def apply_shape_noise(table_s, sigma):

    n_1 = np.random.normal(scale=sigma, size=len(table_s))
    n_2 = np.random.normal(scale=sigma, size=len(table_s))

    a_1 = table_s['g_1'] + n_1
    a_2 = table_s['g_2'] + n_2
    a_3 = 1.0 + table_s['g_1'] * n_1 + table_s['g_2'] * n_2
    a_4 = table_s['g_1'] * n_2 - table_s['g_2'] * n_1
    table_s['e_1'] = (a_1 * a_3 + a_2 * a_4) / (a_3 * a_3 + a_4 * a_4)
    table_s['e_2'] = (a_2 * a_3 - a_1 * a_4) / (a_3 * a_3 + a_4 * a_4)

    return table_s


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('buzzard_mock', choices=[0, 11, 3, 4, 5, 6, 7, 8, 9],
                        help='which Buzzard mock to process', type=int)
    args = parser.parse_args()

    global BUZZARD_PATH
    BUZZARD_PATH = (
        Path(os.getenv('CFS')) / 'desi' / 'mocks' / 'buzzard' /
        'buzzard_v2.0' / 'buzzard-{}'.format(args.buzzard_mock) /
        'addgalspostprocess')

    global TABLE_S, TABLE_C
    for survey in ['des', 'hsc', 'kids']:
        TABLE_S[survey] = read_real_source_catalog(survey)
        TABLE_C[survey] = read_real_calibration_catalog(survey)

    pixel_all = [int(str(fname.stem).split('.')[-1]) for fname in
                 (BUZZARD_PATH / 'truth').iterdir()]

    global TABLE_R
    for survey in ['bgs', 'lrg']:
        table_r = read_random_catalog(survey)
        nside = 8
        pixel = hp.ang2pix(nside, table_r['ra'], table_r['dec'], nest=True,
                           lonlat=True)
        table_r = table_r[np.isin(pixel, pixel_all)]
        table_r = table_r[np.random.random(len(table_r)) < 0.1]
        TABLE_R[survey] = table_r

    global TABLE_IA
    if TABLE_IA is None:
        print('Reading in the IA map...')
        TABLE_IA = Table()
        cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
        z = np.linspace(0, 4, 1000)
        z = interp1d(cosmo.comoving_distance(z), z, kind='cubic')
        TABLE_IA.meta['z_bins'] = z(np.arange(81) * 50)
        nside = 2048
        ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)),
                             lonlat=True)

        # Retrieve IA information for all pixels we selected and their
        # neighbours.
        pixel_ia = np.concatenate(
            [hp.pixelfunc.get_all_neighbours(8, p, nest=True) for p in
             pixel_all])
        pixel_ia = np.unique(pixel_ia)

        # Calculate the nside=2048 pixels that correspond to the nside=8 pixels
        # calculated above.
        ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)),
                             lonlat=True)
        nside = 8
        select = np.isin(hp.pixelfunc.ang2pix(
            nside, ra, dec, nest=True, lonlat=True), pixel_ia)

        # Read the IA data.
        nside = 2048
        TABLE_IA['pix'] = np.arange(hp.nside2npix(nside))[select]
        TABLE_IA['ia_1'] = np.zeros((np.sum(select), 80), dtype=np.float32)
        TABLE_IA['ia_2'] = np.zeros((np.sum(select), 80), dtype=np.float32)

        path = BUZZARD_PATH / 'ia_shear'

        for i in tqdm.tqdm(range(80)):
            fname = 'desi_lensing_shear_buzzard4_{}.fits'.format(i)
            # ia = Table.read(os.path.join(fpath, fname))['T'].data.ravel()
            # TABLE_IA['ia_1'][:, i] = np.real(ia)[select]
            # TABLE_IA['ia_2'][:, i] = np.imag(ia)[select]

    for pixel in tqdm.tqdm(pixel_all):
        table_b = read_buzzard_catalog(pixel)
        table_b = add_ia_information(table_b)

        for survey in ['des', 'hsc', 'kids']:
            table_b['z_' + survey] = photometric_redshift(table_b['z'], survey)

        path = Path('buzzard-{}'.format(args.buzzard_mock)) / 'f_detect.hdf5'

        if not path.isfile():
            table_p = Table()
            table_p['mag_min'] = MAG_BINS[:-1]
            table_p['mag_max'] = MAG_BINS[1:]
            for survey in ['des', 'hsc', 'kids']:
                table_p[survey] = detection_probability(table_b, survey)
            table_p.write(path, path='data')
        else:
            table_p = Table.read(path)

        random = np.random.random(len(table_b))
        for suffix, lensed in zip(['', '_t'], [True, False]):
            for survey in ['des', 'hsc', 'kids']:
                table_b[survey + suffix] = source_target(
                    table_b, table_p, survey, random, lensed=lensed)
            table_b['bgs' + suffix] = bgs_target(table_b, lensed=lensed)
            table_b['lrg' + suffix] = lrg_target(table_b, lensed=lensed)

        select = np.zeros(len(table_b), dtype=bool)
        for suffix in ['', '_t']:
            for survey in ['bgs', 'lrg', 'des', 'hsc', 'kids']:
                select = select | table_b[survey + suffix]
        table_b = table_b[select]

        fname = 'pixel_{}.hdf5'.format(pixel)
        path = Path('buzzard-{}'.format(args.buzzard_mock)) / fname

        buzzard_columns = ['z', 'mu', 'g_1', 'g_2',
                           'ra', 'dec', 'mag', 'ia_1', 'ia_2']
        table_b[buzzard_columns].write(path, path='buzzard', overwrite=True)

        for survey in ['bgs', 'lrg', 'des', 'hsc', 'kids']:
            table_survey = Table()
            table_survey.meta = table_b.meta
            select = table_b[survey] | table_b[survey + '_t']
            table_survey['id_buzzard'] = np.arange(len(table_b))[select]
            table_survey['target'] = table_b[survey][select]
            table_survey['target_t'] = table_b[survey + '_t'][select]

            if survey not in ['bgs', 'lrg']:

                table_survey['g_1'] = (table_b['g_1'][select] +
                                       table_b['ia_1'][select])
                table_survey['g_2'] = (table_b['g_2'][select] +
                                       table_b['ia_2'][select])
                table_survey['mag'] = table_b['mag'][select]
                table_survey['z'] = table_b['z_' + survey][select]
                table_survey = apply_observed_shear(
                    table_survey, survey=survey)

                if survey in ['des', 'kids']:
                    if survey == 'des':
                        sigma = np.array([0.201, 0.204, 0.195, 0.203])
                    else:
                        sigma = np.array([0.274, 0.271, 0.289, 0.287, 0.301])
                    sigma = sigma[np.digitize(
                        table_survey['z'], SOURCE_Z_BINS[survey]) - 1]
                else:
                    sigma = 1.0 / np.sqrt(table_survey['w'])
                table_survey = apply_shape_noise(table_survey, sigma)
                table_survey.remove_columns(['mag', 'g_1', 'g_2'])

            for key in table_survey.colnames:
                dtype = table_survey[key].dtype
                if np.issubdtype(dtype, np.integer):
                    table_survey[key] = table_survey[key].astype(np.int32)
                elif np.issubdtype(dtype, np.floating):
                    table_survey[key] = table_survey[key].astype(np.float32)

            table_survey.write(path, path=survey, append=True)

        for survey in ['bgs', 'lrg']:
            table_r = TABLE_R[survey]
            nside = 8
            pixel_r = hp.ang2pix(nside, table_r['ra'], table_r['dec'],
                                 nest=True, lonlat=True)
            table_r[pixel_r == pixel].write(path, path=survey + '-r',
                                            append=True)


if __name__ == "__main__":
    main()
