import os
import glob
import zebu
import numpy as np
import argparse
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
from scipy.interpolate import interp1d
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, compress_jackknife_fields
from dsigma.physics import critical_surface_density

parser = argparse.ArgumentParser(
    description='Perform the jackknife compression for the lensing ' +
    'measurements of the mock0 challenge.')
parser.add_argument('--zspec', action='store_true',
                    help='use spectroscopic instead of photometric redshfits')
parser.add_argument('--gamma', action='store_true',
                    help='use noise-free shapes')
parser.add_argument('--survey', help='the lens survey', required=True)
parser.add_argument('--zspec_zphot_sys_weights',  action='store_true',
                    help="whether to correct for different redshift weights " +
                    "when using spec-z's")
args = parser.parse_args()


def zspec_systematic_weights(lens_bin, source_bin):

    def total_weight(z_lens, d_com_lens, z_source, d_com_source):

        sigma_crit = critical_surface_density(
            z_lens, z_source, d_l=d_com_lens, d_s=d_com_source)

        return np.sum(sigma_crit**-2)

    z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.5]

    cols_c = ['z_spec', 'z_phot']
    table_c = Table.read(zebu.raw_data_path('calibration', source_bin, 0),
                         format='ascii', data_start=1, names=cols_c)

    cosmo = FlatLambdaCDM(Om0=0.286, H0=100)
    table_c['d_com_phot'] = cosmo.comoving_distance(table_c['z_phot']).value
    table_c['d_com_spec'] = cosmo.comoving_distance(table_c['z_spec']).value

    z_lens = np.linspace(z_bins[lens_bin], z_bins[lens_bin + 1], 500)
    weight_phot_z = np.zeros_like(z_lens)
    weight_spec_z = np.zeros_like(z_lens)

    for i, z in enumerate(z_lens):
        weight_phot_z[i] = total_weight(
            z, cosmo.comoving_distance(z).value, table_c['z_phot'],
            table_c['d_com_phot'])
        weight_spec_z[i] = total_weight(
            z, cosmo.comoving_distance(z).value, table_c['z_spec'],
            table_c['d_com_spec'])

    return interp1d(z_lens, weight_phot_z / weight_spec_z)


if not os.path.exists('jackknife'):
    os.makedirs('jackknife')

try:
    centers = np.genfromtxt(os.path.join('jackknife', 'centers.csv'))
except OSError:
    table_l = zebu.read_raw_data(1, 'random', 3)
    table_l = add_continous_fields(table_l, distance_threshold=1)
    centers = jackknife_field_centers(table_l, n_jk=100)
    np.savetxt(os.path.join('jackknife', 'centers.csv'), centers)

for lens in ['l', 'r']:
    for lens_bin in range(4):
        for source_bin in range(4 if args.survey != 'kids' else 5):
            print('{}: Lens-Bin {}, Source-Bin {}'.format(
                'Lenses' if lens == 'l' else 'Randoms', lens_bin, source_bin))

            fname_base = 'l{}_s{}_{}_{}'.format(
                lens_bin, source_bin, args.survey, lens[0])

            if args.gamma:
                fname_base = fname_base + '_gamma'
            if args.zspec:
                fname_base = fname_base + '_zspec'

            if len(glob.glob(os.path.join('precompute',
                                          fname_base + '*'))) == 0:
                continue

            table_l = []

            for fname in glob.glob(
                    os.path.join('precompute', fname_base + '*')):

                print(fname)

                table_l_i = Table.read(fname, path='data')

                if args.zspec and args.zspec_zphot_sys_weights:
                    table_l_i['w_sys'] = zspec_systematic_weights(
                        lens_bin, source_bin)(table_l_i['z'])

                add_jackknife_fields(table_l_i, centers)
                table_l_i = compress_jackknife_fields(table_l_i)
                table_l.append(table_l_i)

            table_l = compress_jackknife_fields(vstack(table_l))
            # undo vstack concatenate
            table_l.meta['rp_bins'] = table_l_i.meta['rp_bins']

            if args.zspec and args.zspec_zphot_sys_weights:
                fname_base = fname_base + '_w_sys'

            table_l.write(os.path.join('jackknife', fname_base + '.hdf5'),
                          path='data', overwrite=True, serialize_meta=True)
