import os
import zebu
import argparse
import numpy as np
import multiprocessing
from scipy.interpolate import interp1d
from dsigma.physics import critical_surface_density
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, compress_jackknife_fields

# %%

parser = argparse.ArgumentParser(description='Perform the pre-computation ' +
                                 'for the lensing measurements of the mock0 ' +
                                 'challenge.')
parser.add_argument('lens_bin', type=int, help='the tomographic lens bin')
parser.add_argument('source_bin', type=int, help='the tomographic source bin')
parser.add_argument('survey', help='the lens survey')
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--gamma', action='store_true',
                    help='use noise-free shapes')
parser.add_argument('--equal', action='store_true',
                    help='whether to use equal lens weighting')
args = parser.parse_args()

# %%

output_directory = 'precompute'

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

try:
    centers = np.genfromtxt(os.path.join(output_directory, 'centers.csv'))
except OSError:
    table_l = zebu.read_raw_data(1, 'random', 3)
    table_l = add_continous_fields(table_l, distance_threshold=1)
    centers = jackknife_field_centers(table_l, n_jk=100)
    np.savetxt(os.path.join(output_directory, 'centers.csv'), centers)

table_c = zebu.read_raw_data(1, 'calibration', args.source_bin,
                             survey=args.survey)
table_s = zebu.read_raw_data(1, 'source', args.source_bin, survey=args.survey)

if args.gamma:
    table_s['e_1'] = table_s['g_1']
    table_s['e_2'] = table_s['g_2']

if args.zspec:
    table_c['z'] = table_c['z_true']
    table_s['z'] = table_s['z_true']

for table in [table_s, table_c]:
    table = add_maximum_lens_redshift(table, dz_min=0.2, z_err_factor=0)

if args.equal:
    z_lens = np.linspace(zebu.lens_z_bins[args.lens_bin] - 1e-6,
                         zebu.lens_z_bins[args.lens_bin + 1] + 1e-6, 1000)

    weight = np.zeros_like(z_lens)

    for i in range(len(z_lens)):
        sigma_crit = critical_surface_density(z_lens[i], table_c['z'],
                                              cosmology=zebu.cosmo)
        weight[i] = np.sum(table_c['w'] * table_c['w_sys'] * sigma_crit**-2)

    if np.amax(weight) > 0:
        weight = weight / np.amax(weight)

    # If correcting for unequal weight distorts weighting dramatically, don't
    # attempt any re-weighting.
    if np.amin(weight) < 1e-2:
        w_sys = interp1d(z_lens, np.zeros_like(z_lens))
    else:
        w_sys = interp1d(z_lens, 1.0 / weight)

for catalog_type in ['lens', 'random']:

    table_l = zebu.read_raw_data(1, catalog_type, args.lens_bin)
    if args.equal:
        table_l['w_sys'] = w_sys(table_l['z'])
    add_jackknife_fields(table_l, centers)

    output = os.path.join(output_directory, 'l{}_s{}_{}_{}'.format(
        args.lens_bin, args.source_bin, args.survey, catalog_type[0]))

    if args.gamma:
        output = output + '_gamma'
    if args.zspec:
        output = output + '_zspec'
    if args.equal:
        output = output + '_equal'

    output = output + '.hdf5'

    table_l = compress_jackknife_fields(precompute_catalog(
        table_l, table_s, zebu.rp_bins, cosmology=zebu.cosmo, table_c=table_c,
        n_jobs=multiprocessing.cpu_count()))

    table_l.write(output, path='data', overwrite=True)
