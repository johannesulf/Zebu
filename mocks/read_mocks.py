import argparse
import numpy as np
import os
import warnings

from astropy import units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, vstack
from astropy_healpix import HEALPix
from pathlib import Path

SOURCE_Z_BINS = {
    'des': np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
    'hsc': np.array([0.3, 0.6, 0.9, 1.2, 1.5]),
    'kids': np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])}


def random_ra_dec(n, pixels):
    """
    Compute random angular coordinates.

    The function is hard-coded to only return coordinates from the quarter of
    the sky covered by the Buzzard mocks.

    Attributes
    ----------
    n : float
        The density in deg^-2.
    pixels : list
        List of Healpix pixels to cover.

    Returns
    -------
    ra, dec : numpy.ndarray
        Angular coordinates.

    """
    np.random.seed(0)
    hp = HEALPix(8, order='nested')
    n = int(np.pi * (180.0 / np.pi)**2 * n)
    pos = np.random.normal(size=(n, 3))
    pos = pos / np.linalg.norm(pos, axis=1)[:, np.newaxis]
    ra = np.rad2deg(np.arctan2(pos[:, 1], pos[:, 0])) % 180
    dec = np.abs(np.rad2deg(np.arccos(pos[:, 2])) - 90)
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    mask = np.isin(pix, pixels)
    return ra[mask], dec[mask]


def apply_shape_noise(table_s, sigma):
    """
    Apply shape noise to the e_1 and e_2 components.

    Attributes
    ----------
    table_s : astropy.Table
        Table of sources.
    sigma : numpy.ndarray
        Noise for each source.

    Returns
    -------
    table_s : astropy.Table
        Table of sources.

    """
    n_1 = np.random.normal(scale=sigma, size=len(table_s))
    n_2 = np.random.normal(scale=sigma, size=len(table_s))

    a_1 = table_s['e_1'] + n_1
    a_2 = table_s['e_2'] + n_2
    a_3 = 1.0 + table_s['e_1'] * n_1 + table_s['e_2'] * n_2
    a_4 = table_s['e_1'] * n_2 - table_s['e_2'] * n_1
    table_s['e_1'] = (a_1 * a_3 + a_2 * a_4) / (a_3 * a_3 + a_4 * a_4)
    table_s['e_2'] = (a_2 * a_3 - a_1 * a_4) / (a_3 * a_3 + a_4 * a_4)

    return table_s


def read_mock_catalog(survey, path, pixels, magnification=True,
                      fiber_assignment=False, iip_weights=True,
                      intrinsic_alignment=False, shear_bias=True,
                      shape_noise=False, unlensed_coordinates=False,
                      reduced_shear=True, one_pass=False):
    """
    Read in one or multiple mock catalogs.

    Attributes
    ----------
    survey : string or list
        Mock survey to read. Options are 'bgs', 'bgs-r' (BGS randoms), 'lrg',
        'lrg-r', 'des', 'des-c' (DES calibration sample), 'hsc', 'hsc-c',
        'kids', 'kids-c' and 'other' (other DESI targets, used for fiber
        collisions). You can specify multiple surveys at once by passing a
        list.
    path : pathlib.Path, optional
        Path to the folder in which the mocks are located in.
    pixels : list
        List of Healpix pixels to read.
    magnification : bool or list, optional
        Whether to include magnification effects. If list, specifies whether
        magnification effects are included for each survey. Default is True.
    fiber_assignment : bool, optional
        Whether to include fiber assignment for BGS and LRG. Default is False.
    iip_weights : bool, optional
        Whether to include IIP weights to correct for fiber incompleteness.
    intrinsic_alignment : bool, optional
        Whether to include intrinsic alignment effects on the measured shears.
        Default is False.
    shear_bias : bool, optional
        Whether to include shear bias effects on the measured ellipticities.
        Default is True.
    shape_noise : bool, optional
        Whether to include shape noise on the measured ellipticities. Default
        is False.
    unlensed_coordinates : bool, optional
        Whether to account for the absence of magnification on angular
        coordinates by using systematic weights or by using unlensed
        coordinates.
    reduced_shear : bool, optional
        Whether to use the reduced shear or just the shear for the intrinsic
        gravitational signal. Default is True.
    one_pass: bool, optional
        Whether to only use one pass for the fiber assignment. Default is
        False.

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
        * 'bright': whether the object is in BGS_BRIGHT (BGS catalog only)
        * 'abs_mag_r': absolute rest-frame r-band magnitude (for BGS cuts)

    """
    np.random.seed(0)

    if isinstance(survey, str):
        return_array = False
        survey_list = [survey, ]
    else:
        return_array = True
        survey_list = survey

    if isinstance(magnification, bool):
        magnification_list = [magnification, ] * len(survey_list)
    else:
        magnification_list = magnification

    # Read all necessary tables in the files.
    table_all = {}
    for survey in ['buzzard', ] + survey_list:
        table_all[survey] = []
        if survey == 'other':
            continue
        for p in pixels:
            try:
                # Remove '-c' and '-r' for calibration and random samples.
                table_all[survey].append(Table.read(
                    path / 'pixel_{}.hdf5'.format(p),
                    path=survey.split('-')[0]))
            except FileNotFoundError:
                warnings.warn('Cannot find pixel {:d}.'.format(p),
                              stacklevel=2)

    # Assign properties from the Buzzard table to the survey tables.
    for survey, magnification in zip(survey_list, magnification_list):
        if survey == 'other':
            continue
        for i in range(len(table_all['buzzard'])):
            table = table_all[survey][i]
            table_buzzard = table_all['buzzard'][i][table['id_buzzard']]
            for key in ['ra', 'dec', 'mu', 'g_1', 'g_2', 'ia_1', 'ia_2']:
                table[key] = table_buzzard[key]
            if not magnification and unlensed_coordinates:
                table['ra'] = table_buzzard['ra_t']
                table['dec'] = table_buzzard['dec_t']
            if survey in ['bgs', 'bgs-r']:
                table['abs_mag_r'] = table_buzzard['abs_mag_r']
                if magnification:
                    table['abs_mag_r'] += -2.5 * np.log10(
                        table_buzzard['mu'])
            table['z_true'] = table_buzzard['z']

    # Stack all the tables from the individual files together.
    for survey in survey_list:
        if survey == 'other':
            continue
        meta = table_all[survey][0].meta
        meta['bands'] = meta['bands'].astype('S2')
        if 'area' in table_all[survey][0].meta.keys():
            meta['area'] = np.sum([c.meta['area'] for c in table_all[survey]])
        table_all[survey] = vstack(
            table_all[survey], metadata_conflicts='silent')
        table_all[survey].meta = meta

    for survey in survey_list:
        if survey in ['bgs', 'lrg', 'bgs-r', 'lrg-r']:
            table_all[survey]['w_sys'] = np.ones(len(table_all[survey]))

    for survey, magnification in zip(survey_list, magnification_list):
        if survey in ['bgs-r', 'lrg-r', 'other']:
            continue
        if magnification:
            table_all[survey] = table_all[survey][table_all[survey]['target']]
        else:
            table_all[survey] = table_all[survey][
                table_all[survey]['target_t']]
            if not unlensed_coordinates:
                if survey in ['bgs', 'lrg']:
                    key = 'w_sys'
                else:
                    key = 'w'
                table_all[survey][key] = table_all[survey][key] * \
                    table_all[survey]['mu']
            if survey == 'bgs':
                table_all[survey]['bright'] = table_all[survey]['bright_t']

    for survey in survey_list:
        if survey not in ['des', 'hsc', 'kids', 'des-c', 'hsc-c',
                          'kids-c']:
            continue

        table = table_all[survey]

        if not shear_bias:
            if 'm' in table_all[survey].colnames:
                table['m'] = 0
            if 'e_rms' in table_all[survey].colnames:
                table['e_rms'] = np.sqrt(0.5)
            if 'R_11' in table_all[survey].colnames:
                table['R_11'] = 1
            if 'R_22' in table_all[survey].colnames:
                table['R_22'] = 1
            if 'R_21' in table_all[survey].colnames:
                table['R_21'] = 0
            if 'R_12' in table_all[survey].colnames:
                table['R_12'] = 0
        elif not magnification:
            for key in ['m', 'e_rms', 'R_11', 'R_22', 'R_12', 'R_21']:
                if key in table.colnames and key + '_t' in table.colnames:
                    table[key] = table[key + '_t']

        r = np.ones(len(table))
        if 'm' in table.colnames:
            r *= 1 + table['m']
        if 'R_11' in table.colnames:
            r *= 0.5 * (table['R_11'] + table['R_22'])
        if 'e_rms' in table.colnames:
            r *= 2 * (1 - table['e_rms']**2)

        if survey in ['des-c', 'hsc-c', 'kids-c']:
            table['w_sys'] = r

        table['e_1'] = table['g_1'] * r
        table['e_2'] = table['g_2'] * r
        if intrinsic_alignment:
            table['e_1'] += table['ia_1'] * r
            table['e_2'] += table['ia_2'] * r

        if not reduced_shear:
            table['e_1'] -= table['g_1'] * (table['mu'] - 1) / 2.0 * r
            table['e_2'] -= table['g_2'] * (table['mu'] - 1) / 2.0 * r

        table['e_2'] = - table['e_2']

        if shape_noise:

            if survey in ['des', 'kids']:
                if survey == 'des':
                    sigma = np.array([0.201, 0.204, 0.195, 0.203])
                else:
                    sigma = np.array([0.274, 0.271, 0.289, 0.287, 0.301])
                sigma = sigma[np.digitize(
                    table['z'], SOURCE_Z_BINS[survey]) - 1]
            else:
                sigma = 1.0 / np.sqrt(table['w'])

            table_all[survey] = apply_shape_noise(table, sigma)

    for survey in survey_list:
        if survey in ['bgs', 'lrg', 'bgs-r', 'lrg-r']:
            table_all[survey].rename_column('z_true', 'z')
        if survey in ['des-c', 'hsc-c', 'kids-c']:
            table_all[survey] = table_all[survey][::100]

    if 'other' in survey_list:
        np.random.seed(0)
        table_all['other'] = Table()
        table_all['other']['ra'], table_all['other']['dec'] = random_ra_dec(
            600, pixels)

    for survey in survey_list:
        if survey in ['bgs-r', 'lrg-r']:
            np.random.seed(1)
            n = len(table_all[survey]) / table_all[survey].meta['area']
            ra, dec = random_ra_dec(10 * n, pixels)
            idx = np.random.choice(
                len(table_all[survey]), size=len(ra),
                p=table_all[survey]['w_sys'] / np.sum(
                    table_all[survey]['w_sys']))
            table_all[survey] = table_all[survey][idx]
            table_all[survey]['ra'] = ra
            table_all[survey]['dec'] = dec
            for key in table_all[survey].colnames:
                if key not in ['ra', 'dec', 'z', 'bright', 'abs_mag_r',
                               'w_sys']:
                    table_all[survey].remove_column(key)
        if survey in ['bgs', 'lrg'] and fiber_assignment:
            table = table_all[survey]
            fname = '{}{}_assigned_{}_mocks.fits'.format(
                'withmag' if magnification else 'nomag', '_firstpass' if
                one_pass else '', survey)
            table_f = Table.read(path / fname)
            coord = SkyCoord(table['ra'], table['dec'], unit='deg')
            coord_f = SkyCoord(table_f['ra'], table_f['dec'], unit='deg')
            idx, sep2d, dist3d = match_coordinates_sky(coord, coord_f)
            for key in ['has_fiber', 'BITWEIGHT0', 'BITWEIGHT1']:
                table[key] = table_f[key][idx]
            table['n_obs'] = np.zeros(len(table), dtype=int)
            for i in range(len(table)):
                table['n_obs'][i] = (
                    np.binary_repr(table['BITWEIGHT0'][i],
                                   width=64).count('1') +
                    np.binary_repr(table['BITWEIGHT1'][i],
                                   width=64).count('1'))
                table['has_fiber'][i] = np.binary_repr(
                    table['BITWEIGHT0'][i], width=64)[0] == '1'
            table = table[table['has_fiber']]
            if iip_weights:
                table['w_sys'] *= (table['n_obs'] / 128)**-1
            table_all[survey] = table

    for survey in survey_list:
        columns_keep = ['ra', 'dec', 'e_1', 'e_2', 'z_true', 'z', 'e_rms',
                        'm', 'R_11', 'R_22', 'R_12', 'R_21', 'w', 'w_sys',
                        'g_1', 'g_2', 'ia_1', 'ia_2', 'mu', 'bright',
                        'abs_mag_r']
        for key in table_all[survey].colnames:
            if key not in columns_keep:
                table_all[survey].remove_column(key)
            else:
                table_all[survey][key] = table_all[survey][key].astype(float)

    table_all = [table_all[survey] for survey in survey_list]
    if not return_array:
        table_all = table_all[0]

    return table_all


def main():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Script to produce mock catalogs for the lensing mock ' +
        'challenge.',
        epilog='''
The tables have the following columns (if applicable).
    * 'ra': (lensed) right ascension
    * 'dec': (lensed) declination
    * 'z': measured redshift
    * 'z_true': true redshift
    * 'w_sys': lens or calibration systematic weight
    * 'w': source weight (lensing catalogs only)
    * 'e_1'/'e_2': measured ellipticity components (lensing catalogs only)
    * 'g_1'/'g_2': true gravitational shear components
    * 'ia_1'/'ia_2': IA components
    * 'e_rms'/'m'/'R_11'/'R_22'/'R_21'/'R_12': shear biases (lensing only)
    * 'bright': whether the object is in BGS_BRIGHT (BGS catalog only)
    * 'abs_mag_r': absolute rest-frame r-band magnitude (BGS catalog only)''')
    parser.add_argument(
        'filename',
        help="Filename used for the result. Must contain the word 'SURVEY' " +
        "if more than one survey is simulated. In this case, 'SURVEY' will " +
        "replaced by the specific survey, i.e. 'mock_SURVEY.hdf5' will " +
        "'mock_bgs.hdf5', 'mock_lrg.hdf5' etc.")
    parser.add_argument(
        'surveys', choices=['bgs', 'bgs-r', 'lrg', 'lrg-r', 'des', 'hsc',
                            'kids', 'des-c', 'hsc-c', 'kids-c', 'other'],
        help='The survey(s) to simulate.', nargs='+')
    parser.add_argument(
        '-b', '--buzzard', choices=[0, 3, 4, 5, 6, 7, 8, 9, 11],
        help='The underlying Buzzard mock to use.', default=4, type=int)
    parser.add_argument(
        '-p', '--pixels', help='Text file Buzzard listing pixels to be used ' +
        'in the mock. If not provided, all available pixels are used.')
    parser.add_argument(
        '--magnification', help='Include magnification effects.',
        action='store_true')
    parser.add_argument(
        '--unlensed_coordinates', help='If magnification is off, use ' +
        'unlensed coordinates.', action='store_true')
    parser.add_argument(
        '--fiber_assignment', help='Include fiber assignments.',
        action='store_true')
    parser.add_argument(
        '--intrinsic_alignment', help='Include intrinsic alignments.',
        action='store_true')
    parser.add_argument(
        '--shear_bias', help='Include shear bias.', action='store_true')
    parser.add_argument(
        '--shape_noise', help='Include shape noise.', action='store_true')
    parser.add_argument(
        '--one_pass', help='Use one pass for fibers.', action='store_true')
    parser.add_argument(
        '--overwrite', help='Overwrite existing files.', action='store_true')

    args = parser.parse_args()

    if len(args.surveys) > 1 and 'SURVEY' not in args.filename:
        raise ValueError("More than one survey specified but 'SURVEY' not " +
                         "included in the filename.")

    if args.buzzard == 5 and args.intrinsic_alignment:
        raise ValueError("Intrinsic alignment effect not available for " +
                         "Buzzard-5.")

    if len(args.surveys) > 1:
        filenames = [args.filename.replace('SURVEY', survey) for survey in
                     args.surveys]
    else:
        filenames = [args.filename, ]

    path = (Path(os.getenv('CFS')) / 'desicollab' / 'science' / 'c3' /
            'DESI-Lensing' / 'mocks' / 'buzzard-{}'.format(args.buzzard))
    # path = Path('buzzard-4')

    if args.pixels is None:
        pixels = []
        for filepath in path.iterdir():
            if filepath.stem[:6] == 'pixel_':
                pixels.append(int(filepath.stem[6:]))
        pixels = np.array(pixels)
    else:
        pixels = np.atleast_1d(np.genfromtxt(args.pixels, dtype=int))
    pixels = np.sort(pixels)

    print('Buzzard: {}'.format(args.buzzard))
    print('Pixels: {}'.format(', '.join(pixels.astype(str))))
    print('Surveys: {}'.format(', '.join(args.surveys)))
    print('Magnification: {}'.format(args.magnification))
    if not args.magnification:
        print('Unlensed Coordinates: {}'.format(args.unlensed_coordinates))
    print('Fiber Assignment: {}'.format(args.fiber_assignment))
    print('Intrinsic Alignment: {}'.format(args.intrinsic_alignment))
    print('Shear Bias: {}'.format(args.shear_bias))
    print('Shape Noise: {}'.format(args.shape_noise))
    print('Output files: {}'.format(', '.join(filenames)))

    tables = read_mock_catalog(
        args.surveys, path, pixels, magnification=args.magnification,
        fiber_assignment=args.fiber_assignment,
        intrinsic_alignment=args.intrinsic_alignment,
        shear_bias=args.shear_bias, shape_noise=args.shape_noise,
        unlensed_coordinates=args.unlensed_coordinates,
        one_pass=args.one_pass)
    tables = [tables] if not isinstance(tables, list) else tables

    for table, filename, survey in zip(tables, filenames, args.surveys):
        if survey.split('-')[0] in ['des', 'hsc', 'kids']:
            table['z_bin'] = np.digitize(
                table['z'], SOURCE_Z_BINS[survey.split('-')[0]])
        table.write(filename, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
