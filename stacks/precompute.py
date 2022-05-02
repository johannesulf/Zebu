import os
import zebu
import argparse
import numpy as np
import healpy as hp
import multiprocessing
from scipy.interpolate import interp1d
from dsigma.physics import critical_surface_density
from dsigma.precompute import add_maximum_lens_redshift, add_precompute_results
from dsigma.jackknife import compress_jackknife_fields

# This is necessary for lux because otherwise the temporary directory is on
# disk and not in memory.
multiprocessing.process.current_process()._config['tempdir'] = '/dev/shm'

# %%

parser = argparse.ArgumentParser(description='Perform the pre-computation ' +
                                 'for the lensing measurements of the mock0 ' +
                                 'challenge.')
parser.add_argument('stage', type=int, help='stage of the analysis')
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--noisy', action='store_true',
                    help='use noisy shapes')
parser.add_argument('--runit', action='store_true',
                    help='use shapes w/o response bias, i.e. unit response')
parser.add_argument('--noiip', action='store_true',
                    help='ignore IIP systematic weights (for stage >= 4)')
parser.add_argument('--random', action='store_true',
                    help='whether to calculate randoms instead of lenses')
args = parser.parse_args()

# %%

directory = 'precompute'

if not os.path.exists(directory):
    os.makedirs(directory)

# %%

source_magnification = args.stage >= 2
lens_magnification = args.stage >= 3
fiber_assignment = args.stage >= 4
intrinsic_alignment = args.stage >= 5

if args.stage == 0:
    survey_list = ['gen']
else:
    survey_list = ['des', 'hsc', 'kids']

# %%

for survey in survey_list:
    for lens_bin in range(3, -1, -1):

        table_c = zebu.read_mock_data(
            'calibration', 'all', survey=survey,
            magnification=source_magnification)
        table_s = zebu.read_mock_data(
            'source', 'all', survey=survey,
            magnification=source_magnification,
            intrinsic_alignment=intrinsic_alignment)

        if not args.noisy:
            table_s['e_1'] = table_s['g_1']
            table_s['e_2'] = table_s['g_2']

        if args.runit:
            if survey == 'des':
                table_s['e_1'] /= 0.5 * (table_s['R_11'] + table_s['R_22'])
                table_s['e_2'] /= 0.5 * (table_s['R_11'] + table_s['R_22'])
                table_s['R_11'] = 1.0
                table_s['R_22'] = 1.0
            else:
                table_s['e_1'] /= 1 + table_s['m']
                table_s['e_2'] /= 1 + table_s['m']
                table_s['m'] = 0
                if survey == 'hsc':
                    table_s['e_1'] /= 2 * (1 - table_s['e_rms']**2)
                    table_s['e_2'] /= 2 * (1 - table_s['e_rms']**2)
                    table_s['e_rms'] = np.sqrt(0.5)
            table_c['w_sys'] = 1.0

        if args.zspec:
            table_c['z'] = table_c['z_true']
            table_s['z'] = table_s['z_true']

        for table in [table_s, table_c]:
            add_maximum_lens_redshift(table, dz_min=0.2)

        z_lens = np.linspace(zebu.lens_z_bins[lens_bin] - 1e-6,
                             zebu.lens_z_bins[lens_bin + 1] + 1e-6, 30)

        weight = np.zeros_like(z_lens)

        for i in range(len(z_lens)):
            sigma_crit = critical_surface_density(
                z_lens[i], table_c['z'], zebu.cosmo, comoving=True)
            weight[i] = np.sum(
                table_c['w'] * table_c['w_sys'] * sigma_crit**-2 *
                (z_lens[i] < table_c['z_l_max']))

        weight = weight / np.amax(weight)
        w_sys = interp1d(z_lens, 1.0 / weight, kind='cubic')

        if not args.random:
            table_l = zebu.read_mock_data(
                'lens', lens_bin, magnification=lens_magnification,
                fiber_assignment=fiber_assignment)
        else:
            table_l = zebu.read_mock_data(
                'random', lens_bin, magnification=lens_magnification,
                fiber_assignment=fiber_assignment)[::10]
        if args.stage >= 4 and args.noiip:
            table_l['w_sys'] = 1.0
        table_l['w_sys'] *= w_sys(table_l['z'])

        table_l = table_l[table_l['z'] < np.amax(table_c['z_l_max'])]

        table_l['field_jk'] = hp.ang2pix(
            8, table_l['ra'], table_l['dec'], nest=True, lonlat=True)

        fname = 'l{}_{}'.format(lens_bin, survey)

        if args.noisy:
            fname = fname + '_noisy'
        if args.zspec:
            fname = fname + '_zspec'
        if args.runit:
            fname = fname + '_runit'
        if not source_magnification and not lens_magnification:
            fname = fname + '_nomag'
        if source_magnification and not lens_magnification:
            fname = fname + '_nolmag'
        if not fiber_assignment:
            fname = fname + '_nofib'
        if args.noiip:
            fname = fname + '_noiip'

        fname = fname + '.hdf5'

        kwargs = {'cosmology': zebu.cosmo, 'table_c': table_c,
                  'n_jobs': multiprocessing.cpu_count(), 'progress_bar': False}

        add_precompute_results(table_l, table_s, zebu.rp_bins, **kwargs)
        table_l = compress_jackknife_fields(table_l)

        table_l.write(os.path.join(directory, fname),
                      path='lens' if not args.random else 'random',
                      overwrite=True, append=True)

print('Finished successfully!')
