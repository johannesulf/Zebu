import os
import argparse
import numpy as np
import healpy as hp
from scipy.spatial import cKDTree
from scipy.interpolate import splev, splrep
from astropy.table import Table, vstack

z_source_bins = {
    'generic': [0.5, 0.7, 0.9, 1.1, 1.5],
    'des': [0.2, 0.43, 0.63, 0.9, 1.3],
    'hsc': [0.3, 0.6, 0.9, 1.2, 1.5],
    'kids': [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]}


def main(args):

    output = 'region_{}'.format(args.region)

    if not os.path.isdir(output):
        os.makedirs(output)

    print('Reading raw buzzard catalog...')
    nside = 8
    pixel = np.arange(hp.nside2npix(nside))
    ra_pixel, dec_pixel = hp.pix2ang(nside, pixel, nest=True, lonlat=True)
    pixel_use = pixel[ra_dec_in_region(ra_pixel, dec_pixel, args.region)]

    table_b = Table()
    for pixel in pixel_use:
        table_b = vstack([table_b, read_buzzard_catalog(
            pixel, mag_lensed=(args.stage == 2))])
    table_b.meta['area'] = hp.nside2pixarea(
        nside, degrees=True) * len(pixel_use)
    table_b.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
    np.random.seed(0)
    table_b['random_1'] = np.random.random(size=len(table_b))
    table_b['random_2'] = np.random.random(size=len(table_b))
    table_b['randint'] = np.random.randint(3, size=len(table_b))

    if args.stage == 0:

        sample = ['BGS', 'BGS', 'LRG', 'LRG']
        z_min = [0.1, 0.3, 0.5, 0.7]
        z_max = [0.3, 0.5, 0.7, 0.9]

        for lens_bin in range(4):

            print('Reading lens catalog for z-bin {}...'.format(lens_bin))
            table_l = table_b[is_BGS(table_b)]
            table_l.rename_column('z_true', 'z')
            table_l = table_l[(z_min[lens_bin] <= table_l['z']) &
                              (table_l['z'] < z_max[lens_bin])]
            print('Writing lens catalog for z-bin {}...'.format(lens_bin))
            table_l.write(os.path.join(output, 'l{}_nofib.hdf5'.format(
                lens_bin)), overwrite=args.overwrite, path='catalog',
                serialize_meta=True)

            print('Reading random catalog for z-bin {}...'.format(lens_bin))
            table_r = read_random_catalog(args.region, sample[lens_bin])
            table_r = table_r[(z_min[lens_bin] <= table_r['z']) &
                              (table_r['z'] < z_max[lens_bin])]
            print('Writing random catalog for z-bin {}...'.format(lens_bin))
            table_r.write(os.path.join(output, 'r{}.hdf5'.format(lens_bin)),
                          overwrite=args.overwrite, path='catalog')

    if args.stage in [0, 1, 2]:

        if args.stage == 0:
            print('Making tailored source catalog...')
            table_s = subsample_source_catalog(table_b)
            table_s = apply_observed_shear(table_s)
            table_s = apply_shape_noise(table_s, 0.28)
            table_s = apply_photometric_redshift(table_s, None)

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
                                    serialize_meta=True,
                                    path='catalog')
                table_c = table_s_z_bin[np.random.randint(len(table_s_z_bin),
                                                          size=100000)]
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
                table_s = apply_photometric_redshift(
                    table_b, table_c_ref)

                print('Downsampling to target density...')
                table_s = subsample_source_catalog(
                    table_s, table_s_ref=table_s_ref, survey=survey)

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

                for source_bin in range(len(z_bins) - 1):

                    print('Writing source catalog for z-bin {}...'.format(
                        source_bin))
                    use = ((z_bins[source_bin] <= table_s['z']) &
                           (table_s['z'] < z_bins[source_bin + 1]))
                    table_s_z_bin = table_s[use]
                    table_s_z_bin.write(os.path.join(
                        output, 's{}_{}{}.hdf5'.format(
                            source_bin, survey, '_nomag' if args.stage == 1
                            else '')), overwrite=args.overwrite,
                                               path='catalog',
                                               serialize_meta=True)
                    table_c = table_s_z_bin[
                        np.random.randint(len(table_s_z_bin), size=1000000)]
                    table_c.meta = {'bands': table_c.meta['bands']}
                    table_c.write(os.path.join(output, 'c{}_{}{}.hdf5'.format(
                        source_bin, survey, '_nomag' if args.stage == 1 else
                        '')), overwrite=args.overwrite, path='catalog',
                                  serialize_meta=True)

    print('Finished!')

    return


def read_buzzard_catalog(pixel, mag_lensed=False, coord_lensed=False):

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                        'buzzard', 'buzzard_v2.0', 'buzzard-4',
                        'i25_lensing_cat', '8')

    path = os.path.join(path, '{}'.format(pixel // 100), '{}'.format(pixel))
    fname = 'Buzzard_v2.0_lensed-8-{}.fits'.format(pixel)

    table = Table.read(os.path.join(path, fname))
    table.rename_column('GAMMA1', 'gamma_1')
    table.rename_column('GAMMA2', 'gamma_2')
    table.rename_column('Z', 'z_true')
    table.rename_column('KAPPA', 'kappa')

    if mag_lensed:
        table['mag'] = np.hstack((table['LMAG'], table['LMAG_WISE']))
    else:
        table['mag'] = np.hstack((table['TMAG'], table['TMAG_WISE']))

    if coord_lensed:
        table.rename_column('RA', 'ra')
        table.rename_column('DEC', 'dec')
    else:
        table['ra'], table['dec'] = hp.vec2ang(
            np.array([table['PX'], table['PY'], table['PZ']]).T,
            lonlat=True)

    table.keep_columns(['z_true', 'ra', 'dec', 'mag', 'gamma_1', 'gamma_2',
                        'kappa'])

    table.meta = {}

    return table


def read_random_catalog(region, sample, magnification=False):

    path = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                        'buzzard', 'buzzard_v2.0', 'buzzard-4',
                        'DESI_tracers_i25')

    if sample == 'BGS':
        z_min, z_max = 0.1, 0.5
    elif sample == 'LRG':
        z_min, z_max = 0.5, 0.9
    else:
        raise RuntimeError('Unknown lens sample {}.'.format(sample))

    table_r = vstack([Table.read(os.path.join(
        path, 'buzzard_{}_rand.{:02}.fits'.format(sample, i + 1))) for i in
        range(9)])

    table_r.rename_column('RA', 'ra')
    table_r.rename_column('DEC', 'dec')
    table_r['z'] = table_r['Z_COSMO'] + table_r['DZ_RSD']
    table_r.keep_columns(['ra', 'dec', 'z'])

    use = ra_dec_in_region(table_r['ra'], table_r['dec'], region)
    use = use & (z_min <= table_r['z']) & (table_r['z'] < z_max)

    table_r.meta = {}

    return table_r[use].filled()


def ra_dec_in_region(ra, dec, region):

    nside = 8
    pix = hp.ang2pix(nside, np.array(ra), np.array(dec), nest=True,
                     lonlat=True)

    pix_reg = [
        [340, 341, 395, 396, 398, 399, 416, 417, 418, 419, 420, 421, 422, 424,
         425, 426, 637, 638, 639],
        [64, 65, 66, 67, 68, 72, 343, 349, 351, 423, 427, 428, 429, 430, 431,
         432, 434, 435, 440],
        [69, 70, 71, 73, 74, 75, 76, 77, 78, 80, 81, 82, 96, 97, 98, 441, 442,
         443, 446],
        [79, 83, 86, 88, 89, 90, 91, 92, 99, 100, 101, 102, 103, 105, 107, 108,
         109, 110, 112],
        [93, 94, 95, 111, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
         123, 124, 125, 126, 127],
        [391, 397, 400, 401, 402, 403, 404, 405, 406, 408, 409, 410, 483, 488,
         489, 490, 701, 702, 703],
        [128, 129, 130, 136, 407, 411, 412, 413, 414, 415, 433, 436, 486, 487,
         491, 492, 493, 494, 498],
        [131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142, 160, 161, 162,
         437, 438, 439, 444, 445],
        [84, 85, 87, 143, 152, 154, 155, 163, 164, 165, 166, 167, 168, 169,
         170, 171, 172, 176, 447],
        [158, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
         187, 188, 189, 190, 191],
        [144, 145, 146, 147, 148, 149, 150, 153, 232, 495, 499, 504, 505, 506,
         507, 508, 509, 510, 511],
        [151, 156, 157, 159, 234, 235, 237, 238, 239, 248, 249, 250, 251, 254],
        [262, 263, 265, 266, 267, 268, 269, 270, 288, 289, 290, 291, 296, 468,
         759, 764, 765, 766, 767],
        [261, 272, 273, 274, 275, 280, 352, 353, 354, 559, 565, 566, 567, 569,
         570, 571, 572, 573, 574],
        [192, 193, 194, 271, 282, 292, 293, 294, 295, 297, 298, 299, 300, 301,
         302, 304, 306, 469, 471],
        [0, 2, 276, 277, 278, 279, 281, 283, 284, 285, 286, 305, 355, 356, 358,
         360, 361, 362, 364, 575],
        [195, 196, 197, 198, 199, 208, 209, 210, 211, 212, 303, 307, 312, 313,
         314, 315, 316, 318, 319],
        [3, 8, 9, 10, 11, 12, 14, 32, 33, 34, 35, 36, 40, 287, 308, 309, 310,
         311, 317]]

    return np.isin(pix, pix_reg[region - 1])


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
        n_t_tot = 10.0 * 60.0**2
        n_t = z**2 * np.exp(-z / z_0)
        n_t = n_t * n_t_tot / np.sum(n_t * np.diff(z_bins)[0])

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

        use_all = np.zeros(len(table_s), dtype=np.bool)

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
            table_s['g_1'] *= 1 - table_s['e_rms']**2
            table_s['g_2'] *= 1 - table_s['e_rms']**2

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

    table_s = table_s[(0.0 <= table_s['z_true']) & (table_s['z_true'] < 2.0)]
    table_s['z'] = np.zeros(len(table_s)) - 99

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

    table_s.remove_column('random_2')

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
    bgs = table['mag'][:, table.meta['bands'].index('r')] < 20
    return bgs


def is_LRG(table):

    g = table['mag'][:, table.meta['bands'].index('g')]
    r = table['mag'][:, table.meta['bands'].index('r')]
    z = table['mag'][:, table.meta['bands'].index('z')]
    w1 = table['mag'][:, table.meta['bands'].index('w1')]

    lrg = np.ones(len(g), dtype=bool)

    lrg &= z < 20.4
    lrg &= z > 18.0
    lrg &= r - z < 2.5
    lrg &= r - z > 0.8
    lrg &= 1.7 * (r - z) - (r - w1) < 0.6
    lrg &= 1.7 * (r - z) - (r - w1) > -1.0
    lrg &= z < 17.05 + 2 * (r - z)
    lrg &= z > 15.00 + 2 * (r - z)
    lrg &= (z < r - 1.2) | (r < g - 1.7)

    return lrg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build mock challenge')
    parser.add_argument('stage', help='stage of the survey', type=int)
    parser.add_argument('--region', help='region of the sky', type=int,
                        default=1)
    parser.add_argument('--overwrite', help='overwrite existing data',
                        action='store_true')
    args = parser.parse_args()
    main(args)
