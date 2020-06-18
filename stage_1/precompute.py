import os
import zebu
import argparse
import multiprocessing
from dsigma.precompute import add_maximum_lens_redshift, precompute_catalog

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

table_c = zebu.read_raw_data(1, 'calibration', args.source_bin,
                             survey=args.survey)
table_s = zebu.read_raw_data(1, 'source', args.source_bin, survey=args.survey)

if args.gamma:
    table_s['e_1'] = table_s['gamma_1']
    table_s['e_2'] = table_s['gamma_2']

if args.zspec:
    table_c['z'] = table_c['z_true']
    table_s['z'] = table_s['z_true']

for table in [table_s, table_c]:
    table = add_maximum_lens_redshift(table, dz_min=0.0, z_err_factor=0)

for catalog_type in ['lens', 'random']:

    table_l = zebu.read_raw_data(1, catalog_type, args.lens_bin)

    output = os.path.join('precompute', 'l{}_s{}_{}_{}'.format(
        args.lens_bin, args.source_bin, args.survey, catalog_type[0]))

    if args.gamma:
        output = output + '_gamma'
    if args.zspec:
        output = output + '_zspec'

    d_i = 100000

    for i in range(len(table_l) // d_i + 1):

        output_i = output + '_{}.hdf5'.format(i)

        if not os.path.isfile(output_i):
            table_l_i = precompute_catalog(
                table_l[i*d_i:(i+1)*d_i], table_s, zebu.rp_bins,
                cosmology=zebu.cosmo, table_c=table_c,
                n_jobs=multiprocessing.cpu_count())
            table_l_i.write(output_i, path='data', overwrite=True,
                            serialize_meta=True)
