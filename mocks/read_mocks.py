import argparse
import numpy as np
import os
import warnings

from astropy import units as u
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
    hp = HEALPix(8, order='nested')
    n = int(4 * np.pi * (180.0 / np.pi)**2 * n)
    pos = np.random.normal(size=(n, 3))
    pos = pos / np.linalg.norm(pos, axis=1)[:, np.newaxis]
    ra = np.rad2deg(np.arctan2(pos[:, 1], pos[:, 0]))
    dec = np.rad2deg(np.arccos(pos[:, 2])) - 90
    pix = hp.lonlat_to_healpix(ra * u.deg, dec * u.deg)
    mask = np.isin(pix, pixels)
    return ra[mask], dec[mask]


def read_mock_catalog(survey, path, pixels, magnification=True,
                      fiber_assignment=False, intrinsic_alignment=False,
                      shear_bias=True, shape_noise=False):
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
    for survey in survey_list:
        if survey == 'other':
            continue
        for i in range(len(table_all['buzzard'])):
            table_survey = table_all[survey][i]
            table_buzzard = table_all['buzzard'][i][table_survey['id_buzzard']]
            for key in ['ra', 'dec', 'mu', 'g_1', 'g_2', 'ia_1', 'ia_2']:
                table_survey[key] = table_buzzard[key]
            if survey in ['bgs', 'bgs-r']:
                table_survey['abs_mag_r'] = table_buzzard['abs_mag_r']
            table_survey['z_true'] = table_buzzard['z']

    # Stack all the tables from the individual files together.
    for survey in survey_list:
        if survey == 'other':
            continue
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
        if survey in ['bgs-r', 'lrg-r', 'other']:
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
            if survey == 'bgs':
                table_all[survey]['bright'] = table_all[survey]['bright_t']

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
                if key not in ['ra', 'dec', 'z', 'bright', 'abs_mag_r', 'w_sys']:
                    table_all[survey].remove_column(key)

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
    if isinstance(survey_list, str):
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
    * 'z_true': true cosmological redshift
    * 'w_sys': lens or calibration systematic weight
    * 'w': source weight
    * 'e_1'/'e_2': ellipticity components
    * 'g_1'/'g_2': true shear components
    * 'ia_1'/'ia_2': IA components
    * 'e_rms'/'m'/'R_11'/'R_22'/'R_21'/'R_12': shear biases
    * 'bright': whether the object is in BGS_BRIGHT (BGS catalog only)
    * 'abs_mag_r': absolute rest-frame r-band magnitude (for BGS cuts)''')
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
        '--overwrite', help='Overwrite existing files.', action='store_true')

    args = parser.parse_args()

    if len(args.surveys) > 1 and 'SURVEY' not in args.filename:
        raise ValueError("More than one survey specified but 'SURVEY' not " +
                         "included in the filename.")

    if args.fiber_assignment:
        raise ValueError("Fiber assignments haven't been implemented, yet.")

    if len(args.surveys) > 1:
        filenames = [args.filename.replace('SURVEY', survey) for survey in
                     args.surveys]
    else:
        filenames = [args.filename, ]

    path = (Path(os.getenv('CFS')) / 'desi' / 'users' / 'julange' /
            'Zebu' / 'mocks' / 'buzzard-{}'.format(args.buzzard))

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
    print('Fiber Assignment: {}'.format(args.fiber_assignment))
    print('Intrinsic Alignment: {}'.format(args.intrinsic_alignment))
    print('Shear Bias: {}'.format(args.shear_bias))
    print('Shape Noise: {}'.format(args.shape_noise))
    print('Output files: {}'.format(', '.join(filenames)))

    tables = read_mock_catalog(
        args.surveys, path, pixels, magnification=args.magnification,
        fiber_assignment=args.fiber_assignment,
        intrinsic_alignment=args.intrinsic_alignment,
        shear_bias=args.shear_bias, shape_noise=args.shape_noise)
    tables = [tables] if not isinstance(tables, list) else tables

    for table, filename, survey in zip(tables, filenames, args.surveys):
        if survey.split('-')[0] in ['des', 'hsc', 'kids']:
            table['z_bin'] = np.digitize(
                table['z'], SOURCE_Z_BINS[survey.split('-')[0]])
        table.write(filename, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
