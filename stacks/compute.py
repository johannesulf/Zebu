import argparse
import multiprocessing
import numpy as np
from time import time
import zebu

from astropy import units as u
from astropy.io.ascii import convert_numpy
from astropy.table import Table
from astropy_healpix import HEALPix
from dsigma.jackknife import compress_jackknife_fields
from dsigma.precompute import precompute
from pathlib import Path

t_start = time()

# This is necessary for lux because otherwise the temporary directory is on
# disk and not in memory.
multiprocessing.process.current_process()._config['tempdir'] = '/dev/shm'

parser = argparse.ArgumentParser()
parser.add_argument('config', type=int, help='configuration number',
                    choices=range(72))
args = parser.parse_args()

path = Path('results', '{}'.format(args.config))
path.mkdir(exist_ok=True, parents=True)

converters = {'*': [convert_numpy(typ) for typ in (int, float, bool, str)]}
config = dict(Table.read('config.csv', converters=converters)[args.config])

table_l_all, table_s_all, table_c_all = zebu.read_mock_catalog(
    [config['lenses'], config['sources'], config['sources'] + '-c'],
    zebu.MOCK_PATH / 'buzzard-4', zebu.PIXELS,
    magnification=[
        config['lens magnification'], config['source magnification'],
        config['source magnification']],
    fiber_assignment=config['fiber assignment'],
    intrinsic_alignment=config['intrinsic alignment'],
    shear_bias=config['shear bias'],
    shape_noise=config['shape noise'])
table_l_all['field_jk'] = HEALPix(8, order='nested').lonlat_to_healpix(
    table_l_all['ra'] * u.deg, table_l_all['dec'] * u.deg)

if config['lenses'] in ['bgs', 'bgs-r']:
    table_l_all = table_l_all[table_l_all['abs_mag_r'] < zebu.ABS_MAG_R_MAX]

if config['lenses'] in ['bgs-r', 'lrg-r']:
    table_l_all = table_l_all[::3]

table_s_all['z_phot'] = table_s_all['z']
table_c_all['z_phot'] = table_c_all['z']
if not config['photometric redshifts']:
    table_s_all['z'] = table_s_all['z_true']
    table_c_all['z'] = table_c_all['z_true']

z_l_bins = zebu.LENS_Z_BINS[config['lenses'].split('-')[0]]
z_s_bins = zebu.SOURCE_Z_BINS[config['sources']]

lens_source_cut = 0.2

for bin_l, (z_l_min, z_l_max) in enumerate(zip(z_l_bins[:-1], z_l_bins[1:])):

    for bin_s, (z_s_min, z_s_max) in enumerate(
            zip(z_s_bins[:-1], z_s_bins[1:])):

        select = (z_l_min <= table_l_all['z']) & (table_l_all['z'] < z_l_max)
        table_l = table_l_all[select]

        select = ((z_s_min <= table_s_all['z_phot']) &
                  (table_s_all['z_phot'] < z_s_max))
        table_s = table_s_all[select]

        kwargs = dict(
            cosmology=zebu.COSMOLOGY, weighting=0, lens_source_cut=None,
            n_jobs=multiprocessing.cpu_count(), progress_bar=False)

        precompute(table_l, table_s, zebu.THETA_BINS, **kwargs)
        table_l = compress_jackknife_fields(table_l)
        table_l.write(path / 'l{}_s{}_gt.hdf5'.format(bin_l, bin_s),
                      path='data', overwrite=True, serialize_meta=True)

        select = ((z_s_min <= table_c_all['z_phot']) &
                  (table_c_all['z_phot'] < z_s_max))
        table_c = table_c_all[select]

        if config['sources'] != 'hsc':
            table_s['z_bin'] = 0

            z_bins = np.linspace(0, 10, 1001)
            table_n = Table()
            table_n['z'] = 0.5 * (z_bins[1:] + z_bins[:-1])
            table_n['n'] = np.atleast_2d(np.histogram(
                table_c['z_true'], weights=table_c['w_sys'] * table_c['w'],
                bins=z_bins)[0]).T

        kwargs = dict(
            cosmology=zebu.COSMOLOGY, n_jobs=multiprocessing.cpu_count(),
            progress_bar=False)

        if config['sources'] == 'hsc':
            kwargs['table_c'] = table_c
            kwargs['lens_source_cut'] = lens_source_cut
        else:
            kwargs['table_n'] = table_n
            kwargs['lens_source_cut'] = None

        precompute(table_l, table_s, zebu.RP_BINS, **kwargs)
        table_l = compress_jackknife_fields(table_l)

        fname = 'l{}_s{}_ds.hdf5'.format(bin_l, bin_s)

        table_l.write(path / fname, path='data', overwrite=True,
                      serialize_meta=True)

t_end = time()
print('Finished in {:.1f} minutes.'.format((t_end - t_start) / 60.0))
