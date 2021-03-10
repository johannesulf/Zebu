import os
import zebu
import argparse
import numpy as np
import multiprocessing
from scipy.interpolate import interp1d
from dsigma.physics import critical_surface_density
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields

# %%

parser = argparse.ArgumentParser(description='Perform the pre-computation ' +
                                 'for the lensing measurements of the mock0 ' +
                                 'challenge.')
parser.add_argument('stage', type=int, help='stage of the analysis')
parser.add_argument('lens_bin', type=int, help='tomographic lens bin')
parser.add_argument('source_bin', type=int, help='tomographic source bin')
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--noisy', action='store_true',
                    help='use noisy shapes')
parser.add_argument('--region', type=int, help='region of the sky', default=1)
args = parser.parse_args()

# %%

output_directory = 'precompute_{}'.format(args.region)

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

try:
    centers = np.genfromtxt(os.path.join(output_directory, 'centers.csv'))
except OSError:
    table_l = zebu.read_mock_data('random', 0)
    table_l = add_continous_fields(table_l, distance_threshold=1)
    centers = jackknife_field_centers(table_l, n_jk=100)
    np.savetxt(os.path.join(output_directory, 'centers.csv'), centers)

# %%

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 0
fiber_assignment = args.stage >= 3

if args.stage == 0:
    survey_list = ['gen']
else:
    if args.source_bin < 4:
        survey_list = ['des', ]#'hsc', 'kids']
    else:
        survey_list = ['kids']

# %%

for survey in survey_list:

    table_c = zebu.read_mock_data(
        'calibration', args.source_bin, survey=survey,
        magnification=source_magnification)[:10000]
    table_s = zebu.read_mock_data(
        'source', args.source_bin, survey=survey,
        magnification=source_magnification)[:10000]

    if not args.noisy:
        table_s['e_1'] = table_s['g_1']
        table_s['e_2'] = table_s['g_2']

    if args.zspec:
        table_c['z'] = table_c['z_true']
        table_s['z'] = table_s['z_true']

    for table in [table_s, table_c]:
        table = add_maximum_lens_redshift(table, dz_min=0.2, z_err_factor=0)

    z_lens = np.linspace(zebu.lens_z_bins[args.lens_bin] - 1e-6,
                         zebu.lens_z_bins[args.lens_bin + 1] + 1e-6, 1000)

    weight = np.zeros_like(z_lens)

    for i in range(len(z_lens)):
        sigma_crit = critical_surface_density(z_lens[i], table_c['z'],
                                              cosmology=zebu.cosmo)
        weight[i] = np.sum(table_c['w'] * table_c['w_sys'] * sigma_crit**-2 *
                           (z_lens[i] < table_c['z_l_max']))

    if np.amax(weight) > 0:
        weight = weight / np.amax(weight)

    # If correcting for unequal weight distorts weighting dramatically, don't
    # attempt any re-weighting and assign zero weights.
    if np.amin(weight) < 1e-2:
        w_sys = interp1d(z_lens, np.zeros_like(z_lens))
    else:
        w_sys = interp1d(z_lens, 1.0 / weight)

    for catalog_type in ['lens', 'random']:

        table_l = zebu.read_mock_data(
            catalog_type, args.lens_bin, magnification=lens_magnification,
            fiber_assignment=fiber_assignment)
        table_l['w_sys'] *= w_sys(table_l['z'])
        add_jackknife_fields(table_l, centers)

        output = os.path.join(output_directory, 'l{}_s{}_{}'.format(
            args.lens_bin, args.source_bin, survey))

        if args.noisy:
            output = output + '_noisy'
        if args.zspec:
            output = output + '_zspec'
        if fiber_assignment:
            output = output + '_fiber'
        if not source_magnification:
            output = output + '_nosmag'
        if not lens_magnification:
            output = output + '_nolmag'

        if np.all(np.isclose(table_l['w_sys'], 0)) or np.all(
                table_l['z'] > np.amax(table_s['z_l_max'])):
            output = output + '.txt'
            fstream = open(output, "w")
            fstream.write("No suitable lens-source pairs!")
            fstream.close()
            continue

        output = output + '.hdf5'

        kwargs = {'cosmology': zebu.cosmo, 'table_c': table_c,
                  'compress_jackknife_fields': True,
                  'n_jobs': multiprocessing.cpu_count()}

        table_l = precompute_catalog(table_l, table_s, zebu.rp_bins, **kwargs)

        table_l.write(output, path=catalog_type,
                      overwrite=(catalog_type == 'lens'),
                      append=(catalog_type == 'random'))

print('Finished successfully!')
