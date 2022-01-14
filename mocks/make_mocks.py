import os
import sys
import fitsio
import argparse
import numpy as np
import healpy as hp
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.interpolate import splev, splrep
from astropy.table import Table, vstack

z_source_bins = {
    'generic': [0.5, 0.7, 0.9, 1.1, 1.5],
    'des': [0.2, 0.43, 0.63, 0.9, 1.3],
    'hsc': [0.3, 0.6, 0.9, 1.2, 1.5],
    'kids': [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]}


def main(args):

    output = 'mocks'

    if not os.path.isdir(output):
        os.makedirs(output)

    if args.stage == 4:

        for sample in ['bgs', 'lrg']:

            table_l = Table.read(os.path.join(
                    output, '{}_nofib.hdf5'.format(sample)))

            table_l.rename_column('ra', 'RA')
            table_l.rename_column('dec', 'DEC')
            table_l.rename_column('z', 'Z')
            table_l.keep_columns(['RA', 'DEC', 'Z'])
            table_l['TARGETID'] = np.arange(len(table_l))
            table_l['NUMOBS_INIT'] = np.zeros(len(table_l), dtype=int)
            table_l['NUMOBS_MORE'] = np.ones(len(table_l), dtype=int)
            table_l['DESI_TARGET'] = 1
            table_l['PRIORITY'] = 3200
            table_l['OBSCONDITIONS'] = 15
            table_l['SUBPRIORITY'] = np.random.random(len(table_l))
            table_l['TARGETID'] = np.arange(len(table_l))
            fname = os.path.join(output, 'targets_{}.fits'.format(sample))
            if not os.path.exists(fname) or args.overwrite:
                table_l.write(fname, overwrite=True)

            fname = os.path.join(output, 'targeted_{}.fits'.format(sample))
            if os.path.exists(fname):
                table_t = Table.read(fname)
                assert np.all(np.diff(table_t['TARGETID']) >= 0)
                bitweight = table_t['BITWEIGHT0']
                n_obs = np.zeros(len(table_t), dtype=int)
                obs = np.zeros(len(table_t), dtype=bool)

                for i in range(64):
                    n_obs += bitweight % 2
                    if i == 0:
                        obs = n_obs == 1
                    bitweight = bitweight // 2

                table_l = Table.read(os.path.join(
                    output, '{}_nofib.hdf5'.format(sample)))
                table_l['w_sys'] = 64.0 / n_obs[obs][:len(table_l)]
                table_l.write(os.path.join(output, '{}.hdf5'.format(sample)),
                              overwrite=args.overwrite)

        sys.exit()

    print('Reading raw buzzard catalog...')
    nside = 8
    pixel = np.arange(hp.nside2npix(nside))
    ra_pixel, dec_pixel = hp.pix2ang(nside, pixel, nest=True, lonlat=True)
    pixel_use = pixel[ra_dec_in_mock(ra_pixel, dec_pixel)]

    table_b = Table()
    for pixel in tqdm(pixel_use):
        table_b = vstack([table_b, read_buzzard_catalog(
            pixel, mag_lensed=(args.stage >= 2))])
    table_b = table_b.filled()
    table_b.meta['area'] = hp.nside2pixarea(
        nside, degrees=True) * len(pixel_use)
    table_b.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
    np.random.seed(0)
    table_b['random_1'] = np.random.random(
        size=len(table_b)).astype(np.float32)
    table_b['random_2'] = np.random.random(
        size=len(table_b)).astype(np.float32)
    table_b['randint'] = np.random.randint(3, size=len(table_b))

    if args.stage in [0, 3]:

        for sample in ['bgs', 'lrg']:

            if sample == 'bgs':
                table_l = table_b[is_BGS(table_b)]
            else:
                table_l = table_b[is_LRG(table_b)]
            table_l.rename_column('z_true', 'z')
            table_l['w_sys'] = 1.0
            if args.stage == 0:
                table_l['w_sys'] = table_l['w_sys'] * table_l['mu']
            print('Writing lens catalog for {}...'.format(sample.upper()))
            table_l.keep_columns(['z', 'ra', 'dec', 'ra_true', 'dec_true',
                                  'mag', 'w_sys'])
            fname = '{}_nofib'.format(sample)
            if args.stage == 0:
                fname = fname + '_nomag'
            fname = fname + '.hdf5'
            for key in table_l.colnames:
                table_l[key] = table_l[key].astype(np.float32)
            table_l.write(os.path.join(output, fname),
                          overwrite=args.overwrite, path='catalog',
                          serialize_meta=True)

            if args.stage != 0:
                continue

            print('Reading random catalog for {}...'.format(sample.upper()))
            table_r = read_random_catalog(sample)
            print('Writing random catalog for {}...'.format(sample.upper()))
            for key in table_r.colnames:
                table_r[key] = table_r[key].astype(np.float32)
            table_r.write(os.path.join(output, '{}_rand.hdf5'.format(sample)),
                          overwrite=args.overwrite, path='catalog')

    table_b.remove_columns(['ra_true', 'dec_true'])

    if args.stage in [0, 1, 2]:

        if args.stage == 0:
            print('Making tailored source catalog...')
            table_s = subsample_source_catalog(table_b)
            table_s.remove_column('random_2')
            table_s = apply_observed_shear(table_s)
            table_s = apply_shape_noise(table_s, 0.28)
            table_s = apply_photometric_redshift(table_s, None)
            table_s['w'] = table_s['mu']

            for key in table_s.colnames:
                table_s[key] = table_s[key].astype(np.float32)

            z_bins = [0.5, 0.7, 0.9, 1.1, 1.5]

            for source_bin in range(len(z_bins) - 1):
                print('Writing source catalog for z-bin {}...'.format(
                    source_bin))
                use = ((z_bins[source_bin] <= table_s['z']) &
                       (table_s['z'] < z_bins[source_bin + 1]))
                table_s_z_bin = table_s[use]
                table_s_z_bin.write(os.path.join(
                    output, 's{}_gen_nomag.hdf5'.format(source_bin)),
                                    overwrite=args.overwrite,
                                    serialize_meta=True, path='catalog')
                table_c = table_s_z_bin[::100]
                table_c.meta = {}
                table_c.write(os.path.join(output, 'c{}_gen_nomag.hdf5'.format(
                    source_bin)), overwrite=args.overwrite, path='catalog',
                    serialize_meta=True)

        else:
            for survey in ['des', 'hsc', 'kids']:

                z_bins = z_source_bins[survey]

                print('Making {}-like source catalog...'.format(survey))

                print('Reading in reference catalogs...')
                table_s_ref = read_real_source_catalog(survey)
                table_c_ref = read_real_calibration_catalog(survey)

                print('Assigning photometric redshifts...')
                table_b = apply_photometric_redshift(table_b, table_c_ref)

                print('Downsampling to target density...')
                table_s = subsample_source_catalog(
                    table_b, table_s_ref=table_s_ref, survey=survey)
                table_s.remove_column('random_2')

                print('Calculating observed shear...')
                table_s = apply_observed_shear(
                    table_s, table_s_ref=table_s_ref, survey=survey)

                print('Applying shape noise...')
                if survey in ['des', 'kids']:

                    if survey == 'des':
                        sigma = np.array([0.26, 0.29, 0.27, 0.29])
                    else:
                        sigma = np.array([0.276, 0.269, 0.290, 0.281, 0.294])

                    sigma = sigma[np.digitize(table_s['z'], z_bins) - 1]

                else:

                    sigma = 1.0 / np.sqrt(table_s['w'])

                table_s = apply_shape_noise(table_s, sigma)

                if args.stage < 2:
                    table_s['w'] = table_s['w'] * table_s['mu']

                for key in table_s.colnames:
                    table_s[key] = table_s[key].astype(np.float32)

                for source_bin in range(len(z_bins) - 1):

                    print('Writing source catalog for z-bin {}...'.format(
                        source_bin))
                    use = ((z_bins[source_bin] <= table_s['z']) &
                           (table_s['z'] < z_bins[source_bin + 1]))
                    table_s_z_bin = table_s[use]
                    fname = 's{}_{}'.format(source_bin, survey)
                    if args.stage == 1:
                        fname = fname + '_nomag'
                    fname = fname + '.hdf5'
                    table_s_z_bin.write(
                        os.path.join(output, fname), overwrite=args.overwrite,
                        path='catalog', serialize_meta=True)
                    fname = 'c' + fname[1:]
                    table_c = table_s_z_bin[::100]
                    table_c.meta = {'bands': table_c.meta['bands']}
                    table_c.write(
                        os.path.join(output, fname), overwrite=args.overwrite,
                        path='catalog', serialize_meta=True)

    print('Finished!')

    return


def read_buzzard_catalog(pixel, mag_lensed=False, coord_lensed=True):

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                        'buzzard', 'buzzard_v2.0', 'buzzard-4',
                        'addgalspostprocess', 'truth')

    fname = 'Chinchilla-4_cam_rs_scat_shift_lensed.{}.fits'.format(pixel)

    table = Table(fitsio.read(os.path.join(path, fname),
                              columns=['GAMMA1', 'GAMMA2', 'Z', 'MU']))
    table.rename_column('GAMMA1', 'gamma_1')
    table.rename_column('GAMMA2', 'gamma_2')
    table.rename_column('Z', 'z_true')
    table.rename_column('MU', 'mu')

    table['ra'] = fitsio.read(
        os.path.join(path, fname), columns=['RA'])['RA']
    table['dec'] = fitsio.read(
        os.path.join(path, fname), columns=['DEC'])['DEC']
    pos = fitsio.read(
        os.path.join(path, fname), columns=['PX', 'PY', 'PZ'])
    table['ra_true'], table['dec_true'] = hp.vec2ang(
        np.array([pos['PX'], pos['PY'], pos['PZ']]).T,
        lonlat=True)

    if mag_lensed:
        s = fitsio.read(os.path.join(path, fname), columns=['SIZE'])['SIZE']
    else:
        s = fitsio.read(os.path.join(path, fname), columns=['TSIZE'])['TSIZE']

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                        'buzzard', 'buzzard_v2.0', 'buzzard-4',
                        'addgalspostprocess', 'surveymags')
    fname = 'Chinchilla-4-aux.{}.fits'.format(pixel)

    if mag_lensed:
        mag = fitsio.read(os.path.join(path, fname), columns=['LMAG'])['LMAG']
    else:
        mag = fitsio.read(os.path.join(path, fname), columns=['TMAG'])['TMAG']

    table['mag'] = mag[:, [1, 2, 3, 4, 5, -2, -1]]

    f_r = 10**((table['mag'][:, 1] - 22.5)/-2.5)
    f_z = 10**((table['mag'][:, 3] - 22.5)/-2.5)

    rf = 0.105
    f_z_fib = f_z / (1 - np.exp(-1)) * (1 - np.exp(-rf / s))
    table['z_fib_mag'] = 22.5 - 2.5 * np.log10(f_z_fib)

    rf = 0.275
    f_r_fib = f_r / (1 - np.exp(-1)) * (1 - np.exp(-rf / s))
    table['r_fib_mag'] = 22.5 - 2.5 * np.log10(f_r_fib)

    table.meta = {}

    for key in table.colnames:
        table[key] = table[key].astype(np.float32)

    table = table[(0.0 <= table['z_true']) & (table['z_true'] < 2.0)]
    table = table[ra_dec_in_mock(table['ra'], table['dec'])]

    return table


def read_random_catalog(sample):

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                        'buzzard', 'buzzard_v2.0', 'buzzard-4',
                        'addgalspostprocess', 'desi_targets')
    fname = '{}_rand.fits'.format(sample)

    table = Table(fitsio.read(os.path.join(path, fname),
                              columns=['ra', 'dec', 'redshift']))

    table.rename_column('redshift', 'z')
    table = table[ra_dec_in_mock(table['ra'], table['dec'])]

    table.meta = {}

    for key in table.colnames:
        table[key] = table[key].astype(np.float32)

    return table


def ra_dec_in_mock(ra, dec):

    nside = 8
    pix = hp.ang2pix(nside, np.array(ra), np.array(dec), nest=True,
                     lonlat=True)

    pix_use = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39,
               48, 49, 50, 51, 52, 53, 54, 55, 69, 70, 71, 73, 74, 75, 76, 77,
               78, 79, 80, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99, 100, 101,
               102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
               115, 120, 121, 122, 123, 373, 374, 375, 377, 378, 379, 380, 381,
               382, 383]

    return np.isin(pix, pix_use)


def subsample_source_catalog(table_s, table_s_ref=None, survey=None):

    if table_s_ref is None:

        # measured redshift distribution
        z_bins = np.linspace(0.0, 2.0, 201)
        table_s = table_s[table_s['z_true'] >= 0]
        table_s = table_s[table_s['z_true'] < 2.0]
        n_s = (np.histogram(table_s['z_true'], bins=z_bins)[0] /
               table_s.meta['area'] / np.diff(z_bins)[0])

        # target redshift distribution
        z = 0.5 * (z_bins[1:] + z_bins[:-1])
        z_0 = 0.2
        n_t_tot = 30.0 * 60.0**2
        n_t = z**2 * np.exp(-z / z_0)
        n_t *= n_t_tot / np.sum(n_t * np.diff(z_bins)[0])

        # downsample input catalog
        z_dig = np.digitize(table_s['z_true'], z_bins) - 1
        table_s = table_s[(n_t / n_s)[z_dig] > np.random.random(
            size=len(table_s))]

    else:

        if survey == 'hsc':
            band = 'i'
        else:
            band = 'r'

        z_bins = z_source_bins[survey]

        mag_bins = np.linspace(18.0, 26.0, 81)
        mag_dig = np.digitize(
            table_s['mag'][:, table_s.meta['bands'].index(band)], mag_bins)
        mag_dig_ref = np.digitize(
            table_s_ref['mag'][:, table_s_ref.meta['bands'].index(band)],
            mag_bins)

        if survey == 'hsc':
            mag_dig[table_s['z'] > 1.2] -= 1

        mag_dig = np.minimum(np.maximum(mag_dig, 0), len(mag_bins) - 2)
        mag_dig_ref = np.minimum(np.maximum(mag_dig_ref, 0), len(mag_bins) - 2)

        use_all = np.zeros(len(table_s), dtype=bool)

        for i in range(len(z_bins) - 1):

            use = ((z_bins[i] <= table_s['z']) &
                   (table_s['z'] < z_bins[i + 1]))
            use_ref = ((z_bins[i] <= table_s_ref['z']) &
                       (table_s_ref['z'] < z_bins[i + 1]))

            n = np.bincount(mag_dig[use], minlength=len(mag_bins) - 1)
            n_ref = np.bincount(
                mag_dig_ref[use_ref], minlength=len(mag_bins) - 1)

            p = ((n_ref[mag_dig[use]] / table_s_ref.meta['area']) /
                 (n[mag_dig[use]] / table_s.meta['area']))
            use_all[use] = p > table_s['random_1'][use]

        table_s = table_s[use_all]

    table_s.remove_column('random_1')

    return table_s


def apply_observed_shear(table_s, table_s_ref=None, survey=None):

    table_s['g_1'] = table_s['gamma_1']
    table_s['g_2'] = table_s['gamma_2']

    if table_s_ref is not None:

        mag = table_s['mag'][:, [b in table_s_ref.meta['bands'] for b in
                                 table_s.meta['bands']]]
        mag_ref = table_s_ref['mag']

        tree = cKDTree(mag_ref)
        idx = tree.query(mag, k=3)[1]
        idx = np.array(idx)
        idx = idx[np.arange(len(idx)), table_s['randint']]

        for key in table_s_ref.colnames:
            if key in ['m', 'w', 'R_11', 'R_22', 'R_12', 'R_21',
                       'e_rms']:
                table_s[key] = table_s_ref[key][idx]

        if survey.lower() == 'kids':
            m = np.array([-0.017, -0.008, -0.015, 0.010, 0.006])
            z_dig = np.digitize(
                table_s['z'], z_source_bins[survey.lower()]) - 1
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

    table_s.remove_column('randint')

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


def apply_photometric_redshift(table_s, table_c_ref):

    table_s['z'] = np.zeros(len(table_s), dtype=np.float32) - 99

    if table_c_ref is None:

        while np.any((0 > table_s['z']) | (table_s['z'] > 2.0)):
            use = (0 <= table_s['z']) & (table_s['z'] <= 2.0)
            table_s['z'] = np.where(
                use, table_s['z'], table_s['z_true'] + np.random.normal(
                    scale=0.1 * (1 + table_s['z_true']), size=len(table_s)))

    else:
        z_bins = np.linspace(0.0, 2.0, 100)
        z_bins_fine = np.linspace(0.0, 2.0, 10000)

        table_c_ref = table_c_ref[(0.0 <= table_c_ref['z_true']) &
                                  (table_c_ref['z_true'] < 2.0)]
        table_c_ref = table_c_ref[(0.0 <= table_c_ref['z']) &
                                  (table_c_ref['z'] < 2.0)]

        for i in range(len(z_bins) - 1):

            use = ((z_bins[i] <= table_s['z_true']) &
                   (table_s['z_true'] < z_bins[i + 1]))
            use_ref = ((z_bins[i] <= table_c_ref['z_true']) &
                       (table_c_ref['z_true'] < z_bins[i + 1]))

            z_hist = np.histogram(table_c_ref['z'][use_ref],
                                  weights=table_c_ref['w_sys'][use_ref],
                                  bins=z_bins)[0]

            cdf_fine = np.cumsum(splev(z_bins_fine, splrep(
                0.5 * (z_bins[1:] + z_bins[:-1]), z_hist)))
            if cdf_fine[-1] == 0:
                continue
            cdf_fine = cdf_fine / cdf_fine[-1]

            z = z_bins_fine[np.searchsorted(
                cdf_fine, table_s['random_2'][use])]

            table_s['z'][use] = z

    return table_s


def read_real_source_catalog(survey):

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'users',
                        'cblake', 'lensing', 'lenscats')

    if survey.lower() in ['kids', 'hsc']:
        fname = '{}_mag.fits'.format(survey.lower())
    elif survey.lower() == 'des':
        fname = 'des_metacal_mag.fits'
    else:
        raise RuntimeError('Unkown survey {}.'.format(survey))

    table_s = Table.read(os.path.join(path, fname))

    if survey == 'des':

        table_s.meta['area'] = 142.9
        table_s.meta['bands'] = 'riz'

        table_s['mag'] = np.zeros((len(table_s), len(table_s.meta['bands'])))
        for i, band in enumerate(table_s.meta['bands']):
            table_s['mag'][:, i] = 30. - 2.5 * np.log10(table_s[
                'flux_{}'.format(band)])
        table_s.rename_column('weight', 'w')
        table_s.rename_column('R11', 'R_11')
        table_s.rename_column('R22', 'R_22')
        table_s['R_12'] = np.zeros(len(table_s))
        table_s['R_21'] = np.zeros(len(table_s))
        table_s.rename_column('zphotmof', 'z')

        table_s.keep_columns(['mag', 'w', 'R_11', 'R_22', 'R_12', 'R_21', 'z'])

        use = (0.2 <= table_s['z']) & (table_s['z'] < 1.3)
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

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'users',
                        'cblake', 'lensing', 'lenscats')

    if survey.lower() in ['kids', 'hsc']:
        fname = '{}_cal.fits'.format(survey.lower())
    elif survey.lower() == 'des':
        fname = 'des_metacal_cal.fits'
    else:
        raise RuntimeError('Unkown survey {}.'.format(survey))

    table_c = Table.read(os.path.join(path, fname))

    if survey == 'des':

        table_c.rename_column('zphot', 'z')
        table_c.rename_column('zmc', 'z_true')
        table_c.rename_column('weinz', 'w_sys')

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


def is_BGS(table):

    g = table['mag'][:, table.meta['bands'].index('g')]
    r = table['mag'][:, table.meta['bands'].index('r')]
    z = table['mag'][:, table.meta['bands'].index('z')]
    w1 = table['mag'][:, table.meta['bands'].index('w1')]
    r_fib = table['r_fib_mag']
    r_off = 0.05
    # BASS r-mag offset with DECaLS.
    offset = 0.04

    bgs = np.ones(len(table), dtype=bool)

    bgs &= g - r > -1
    bgs &= g - r < 4
    bgs &= r - z > -1
    bgs &= r - z < 4

    fmc = np.zeros(len(table), dtype=bool)
    fmc |= ((r_fib < (2.9 + 1.2 + 1.0) + r) & (r < 17.8))
    fmc |= ((r_fib < 22.9) & (r < 20.0) & (r > 17.8))
    fmc |= ((r_fib < 2.9 + r) & (r > 20))

    bgs &= fmc

    is_bright = np.ones(len(table), dtype=bool)
    is_bright &= r < 19.5 + offset + r_off
    is_bright &= r > 12.0 - r_off
    is_bright &= r_fib > 15.0

    is_faint = np.ones(len(table), dtype=bool)
    is_faint &= r < 20.22
    is_faint &= r > 19.5 + offset + r_off

    # D. Schlegel - ChangHoon H. color selection to get a high redshift
    # success rate.
    schlegel_color = (z - w1) - 1.2 * (g - (r - offset)) + 1.2
    is_faint &= (r_fib < 20.75 + offset) | (
        (r_fib < 21.5 + offset) & (schlegel_color > 0.0))

    bgs &= (is_bright | is_faint)

    return bgs


def is_LRG(table):

    g = table['mag'][:, table.meta['bands'].index('g')]
    r = table['mag'][:, table.meta['bands'].index('r')]
    z = table['mag'][:, table.meta['bands'].index('z')]
    w1 = table['mag'][:, table.meta['bands'].index('w1')]
    z_fib = table['z_fib_mag']

    lrg = np.ones(len(table), dtype=bool)

    lrg &= z - w1 > 0.8 * (r - z) - 0.6  # non-stellar cut
    lrg &= z_fib < 21.61   # faint limit
    lrg &= (g - w1 > 2.9) | (r - w1 > 1.8)  # low-z cuts
    lrg &= (((r - w1 > (w1 - 17.14) * 1.8 + 0.4) &
             (r - w1 > (w1 - 16.33) * 1.)) |
            (r - w1 > 3.3 + 0.275))  # double sliding cuts and high-z extension

    return lrg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build mock challenge')
    parser.add_argument('stage', help='stage of the survey', type=int)
    parser.add_argument('--overwrite', help='overwrite existing data',
                        action='store_true')
    args = parser.parse_args()
    main(args)
