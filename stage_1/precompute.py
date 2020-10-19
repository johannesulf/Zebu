import os
import zebu
import argparse
import numpy as np
import multiprocessing
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, compress_jackknife_fields

# %%

parser = argparse.ArgumentParser(description='Perform the pre-computation ' +
                                 'for the lensing measurements of the mock0 ' +
                                 'challenge.')
parser.add_argument('--lens_bin', type=int, help='the tomographic lens bin',
                    required=True)
parser.add_argument('--source_bin', type=int,
                    help='the tomographic source bin', required=True)
parser.add_argument('--survey', help='the lens survey', required=True)
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--gamma', action='store_true',
                    help='use noise-free shapes')
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
    table = add_maximum_lens_redshift(table, dz_min=0.0, z_err_factor=0)

for catalog_type in ['lens', 'random']:

    table_l = zebu.read_raw_data(1, catalog_type, args.lens_bin)
    add_jackknife_fields(table_l, centers)

    output = os.path.join(output_directory, 'l{}_s{}_{}_{}'.format(
        args.lens_bin, args.source_bin, args.survey, catalog_type[0]))

    if args.gamma:
        output = output + '_gamma'
    if args.zspec:
        output = output + '_zspec'

    output = output + '.hdf5'

    table_l = compress_jackknife_fields(precompute_catalog(
        table_l, table_s, zebu.rp_bins, cosmology=zebu.cosmo, table_c=table_c,
        n_jobs=multiprocessing.cpu_count()))

    table_l.write(output, path='data', overwrite=True)
