import zebu
import argparse
import numpy as np
from time import time
import multiprocessing
from pathlib import Path
from astropy import units as u
from astropy.table import Table
from astropy_healpix import HEALPix
from scipy.interpolate import interp1d
from astropy.io.ascii import convert_numpy
from dsigma.physics import critical_surface_density
from dsigma.precompute import precompute
from dsigma.jackknife import compress_jackknife_fields

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
    magnification=[
        config['lens magnification'], config['source magnification'],
        config['source magnification']],
    fiber_assignment=config['fiber assignment'],
    intrinsic_alignment=config['intrinsic alignment'],
    shear_bias=config['shear bias'],
    shape_noise=config['shape noise'])
table_l_all['field_jk'] = HEALPix(8, order='nested').lonlat_to_healpix(
    table_l_all['ra'] * u.deg, table_l_all['dec'] * u.deg)

table_s_all['z_phot'] = table_s_all['z']
table_c_all['z_phot'] = table_c_all['z']
if not config['photometric redshifts']:
    table_s_all['z'] = table_s_all['z_true']
    table_c_all['z'] = table_c_all['z_true']

z_l_bins = zebu.LENS_Z_BINS[config['lenses'].split('-')[0]]
z_s_bins = zebu.SOURCE_Z_BINS[config['sources']]

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

        precompute(table_l, table_s, zebu.RP_BINS, **kwargs)
        table_l = compress_jackknife_fields(table_l)
        table_l.write(path / 'l{}_s{}_gt.hdf5'.format(bin_l, bin_s),
                      path='data', overwrite=True)

lens_source_cut = 0.2

for bin_l, (z_l_min, z_l_max) in enumerate(zip(z_l_bins[:-1], z_l_bins[1:])):

    select = (z_l_min <= table_l_all['z']) & (table_l_all['z'] < z_l_max)
    table_l = table_l_all[select]

    table_s = table_s_all[table_s_all['z'] >=
                          np.amin(table_l['z']) + lens_source_cut]
    table_c = table_c_all[table_c_all['z'] >=
                          np.amin(table_l['z']) + lens_source_cut]

    z_l = np.linspace(z_l_min - 1e-6, z_l_max + 1e-6, 100)
    w_sys_inv = np.zeros_like(z_l)

    for i in range(len(z_l)):
        sigma_crit = critical_surface_density(
            z_l[i], table_c['z'], zebu.COSMOLOGY, comoving=True)
        w_sys_inv[i] = np.sum(
            table_c['w'] * table_c['w_sys'] * sigma_crit**-2 *
            (z_l[i] + lens_source_cut < table_c['z']))

    w_sys_inv = w_sys_inv / np.amax(w_sys_inv)
    w_sys = interp1d(z_l, 1.0 / w_sys_inv, kind='cubic')
    table_l['w_sys'] *= w_sys(table_l['z'])

    kwargs = dict(
        cosmology=zebu.COSMOLOGY, table_c=table_c,
        lens_source_cut=lens_source_cut, n_jobs=multiprocessing.cpu_count(),
        progress_bar=False)

    precompute(table_l, table_s, zebu.RP_BINS, **kwargs)
    table_l = compress_jackknife_fields(table_l)
    table_l.write(path / 'l{}_ds.hdf5'.format(bin_l), path='data',
                  overwrite=True)

t_end = time()
print('Finished in {:.1f} minutes.'.format((t_end - t_start) / 60.0))
