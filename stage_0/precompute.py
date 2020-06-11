import os
import sys
import zebu
import argparse
import multiprocessing
import numpy as np
from astropy.table import Table
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog

# %%

parser = argparse.ArgumentParser(description='Perform the pre-computation ' +
                                 'for the lensing measurements of the mock0 ' +
                                 'challenge.')
parser.add_argument('--lens_bin', type=int, help='the tomographic lens bin',
                    required=True)
parser.add_argument('--source_bin', type=int,
                    help='the tomographic source bin', required=True)
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--gamma', action='store_true',
                    help='use noise-free shapes')
args = parser.parse_args()

if args.lens_bin == 3 and args.source_bin == 0:
    sys.exit(0)

# %%

cols_c = ['z_true', 'z']
cols_l = ['ra', 'dec', 'z', 'w_sys']
cols_s = ['ra', 'dec', 'z_true', 'z', 'gamma_1', 'gamma_2', 'e_1', 'e_2', 'w']

table_c = Table.read(zebu.raw_data_path(0, 'calibration', args.source_bin),
                     format='ascii', data_start=1, names=cols_c)
table_s = Table.read(zebu.raw_data_path(0, 'source', args.source_bin),
                     format='ascii', data_start=1, names=cols_s)

if args.gamma:
    table_s['e_1'] = table_s['gamma_1']
    table_s['e_2'] = table_s['gamma_2']

table_c['w'] = np.ones(len(table_c))
table_c['w_sys'] = np.ones(len(table_c))
table_s['z_err'] = 0.1 * (1 + table_s['z_true'])
table_c['z_err'] = 0.1 * (1 + table_c['z_true'])

if args.zspec:
    table_c['z'] = table_c['z_true']
    table_s['z'] = table_s['z_true']

for table in [table_s, table_c]:
    table = add_maximum_lens_redshift(table, dz_min=0.0, z_err_factor=0)

for catalog_type in ['lens', 'random']:

    table_l = Table.read(zebu.raw_data_path(0, catalog_type, args.lens_bin),
                         format='ascii', data_start=1, names=cols_l)

    table_l = precompute_catalog(
        table_l, table_s, zebu.rp_bins, cosmology=zebu.cosmo,
        table_c=table_c, n_jobs=multiprocessing.cpu_count())

    output = os.path.join('precompute', 'l{}_s{}_{}'.format(
        args.lens_bin, args.source_bin, catalog_type[0]))

    if args.gamma:
        output = output + '_gamma'
    if args.zspec:
        output = output + '_zspec'

    output = output + '.hdf5'

    table_l.write(output, path='data', overwrite=True)
