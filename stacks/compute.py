import os
import zebu
import argparse
import numpy as np
import healpy as hp
from time import time
import multiprocessing
from astropy.table import Table
from scipy.interpolate import interp1d
from astropy.io.ascii import convert_numpy
from dsigma.physics import critical_surface_density
from dsigma.precompute import add_maximum_lens_redshift, add_precompute_results
from dsigma.jackknife import compress_jackknife_fields

t_start = time()

# This is necessary for lux because otherwise the temporary directory is on
# disk and not in memory.
multiprocessing.process.current_process()._config['tempdir'] = '/dev/shm'

parser = argparse.ArgumentParser()
parser.add_argument('config', type=int, help='configuration number',
                    choices=range(72))
args = parser.parse_args()

fpath = os.path.join('results', '{}'.format(args.config))

if not os.path.exists(fpath):
    os.makedirs(fpath)

converters = {'*': [convert_numpy(typ) for typ in (int, float, bool, str)]}
config = dict(Table.read('config.csv', converters=converters)[args.config])

cat_l_all, cat_s_all, cat_c_all = zebu.read_mock_catalog(
    [config['lenses'], config['sources'], config['sources'] + '-c'],
    magnification=[
        config['lens magnification'], config['source magnification'],
        config['source magnification']],
    fiber_assignment=config['fiber assignment'],
    intrinsic_alignment=config['intrinsic alignment'],
    photometric_redshifts=config['photometric redshifts'],
    shear_bias=config['shear bias'],
    shape_noise=config['shape noise'])
cat_l_all['field_jk'] = hp.ang2pix(
    8, cat_l_all['ra'], cat_l_all['dec'], nest=True, lonlat=True)

z_l_bins = zebu.LENS_Z_BINS[config['lenses'].split('-')[0]]
z_s_bins = zebu.SOURCE_Z_BINS[config['sources']]

for bin_l, (z_l_min, z_l_max) in enumerate(zip(z_l_bins[:-1], z_l_bins[1:])):

    for bin_s, (z_s_min, z_s_max) in enumerate(
            zip(z_s_bins[:-1], z_s_bins[1:])):

        select = (z_l_min <= cat_l_all['z']) & (cat_l_all['z'] < z_l_max)
        cat_l = cat_l_all[select]

        select = (z_s_min <= cat_s_all['z']) & (cat_s_all['z'] < z_s_max)
        cat_s = cat_s_all[select]
        add_maximum_lens_redshift(cat_s, dz_min=-99)

        kwargs = {'cosmology': zebu.COSMOLOGY, 'weighting': 0,
                  'n_jobs': multiprocessing.cpu_count(), 'progress_bar': False}

        add_precompute_results(cat_l, cat_s, zebu.RP_BINS, **kwargs)
        cat_l = compress_jackknife_fields(cat_l)
        cat_l.write(
            os.path.join(fpath, 'l{}_s{}_gt.hdf5'.format(bin_l, bin_s)),
            path='data', overwrite=True)

for bin_l, (z_l_min, z_l_max) in enumerate(zip(z_l_bins[:-1], z_l_bins[1:])):

    select = (z_l_min <= cat_l_all['z']) & (cat_l_all['z'] < z_l_max)
    cat_l = cat_l_all[select]

    for cat in [cat_s_all, cat_c_all]:
        add_maximum_lens_redshift(cat, dz_min=0.2)

    cat_s = cat_s_all[cat_s_all['z_l_max'] >= np.amin(cat_l['z'])]
    cat_c = cat_c_all[cat_c_all['z_l_max'] >= np.amin(cat_l['z'])]

    z_l = np.linspace(z_l_min - 1e-6, z_l_max + 1e-6, 100)
    w_sys_inv = np.zeros_like(z_l)

    for i in range(len(z_l)):
        sigma_crit = critical_surface_density(
            z_l[i], cat_c['z'], zebu.COSMOLOGY, comoving=True)
        w_sys_inv[i] = np.sum(cat_c['w'] * cat_c['w_sys'] * sigma_crit**-2 *
                              (z_l[i] < cat_c['z_l_max']))

    w_sys_inv = w_sys_inv / np.amax(w_sys_inv)
    w_sys = interp1d(z_l, 1.0 / w_sys_inv, kind='cubic')
    cat_l['w_sys'] *= w_sys(cat_l['z'])

    kwargs = {'cosmology': zebu.COSMOLOGY, 'table_c': cat_c,
              'n_jobs': multiprocessing.cpu_count(), 'progress_bar': False}

    add_precompute_results(cat_l, cat_s, zebu.RP_BINS, **kwargs)
    cat_l = compress_jackknife_fields(cat_l)
    cat_l.write(os.path.join(fpath, 'l{}_ds.hdf5'.format(bin_l)), path='data',
                overwrite=True)

t_end = time()
print('Finished in {:.1f} minutes.'.format((t_end - t_start) / 60.0))
