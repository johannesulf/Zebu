import os
import argparse
import numpy as np
import multiprocessing
from astropy.table import Table, vstack
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute_catalog, add_maximum_lens_redshift
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from dsigma.surveys import kids
from astropy.cosmology import FlatLambdaCDM

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal for the LWB project.')

parser.add_argument('survey', help='the lens survey')
args = parser.parse_args()

# %%

if args.survey.lower() == 'kids':

    table_s = Table()

    for region in [9, 12, 15, 23, 'S']:
        table_s = vstack([table_s, Table.read(os.path.join(
            'raw', 'KV450_G{}_reweight_3x4x4_v2_good.cat'.format(region)),
                                              hdu=1)],
                         metadata_conflicts='silent')

    table_s = table_s[table_s['MASK'] == 0]
    table_s = dsigma_table(table_s, 'source', survey='KiDS', version='KV450')

    z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]
    table_s['z_bin'] = np.digitize(table_s['z'], z_bins) - 1

    nz = np.array([np.genfromtxt(os.path.join(
        'raw', 'Nz_DIR_z{}t{}.asc'.format(z_min, z_max))).T for
        z_min, z_max in zip(z_bins[:-1], z_bins[1:])])

    table_s = table_s[(table_s['z_bin'] >= 0) &
                      (table_s['z_bin'] < len(z_bins) - 1)]
    table_s = add_maximum_lens_redshift(table_s, dz_min=0.1)
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'],
                                                  version='KV450')


elif args.survey.lower() == 'cfht':

    table_s = Table.read(os.path.join('raw', 'CFHTLS.csv'))
    table_c = table_s[np.random.randint(len(table_s), size=100000)]
    table_s = dsigma_table(table_s, 'source', survey='cfhtls')
    table_s = table_s[table_s['z'] > 0.25 - 1e-6]
    table_s = table_s[table_s['z'] < 1.3 + 1e-6]
    table_s = add_maximum_lens_redshift(table_s, dz_min=0.2)

    z_pdf = np.linspace(0.0, 3.5, 71)
    z_pdf = 0.5 * (z_pdf[1:] + z_pdf[:-1]) + 0.005

    table_c['w_sys'] = np.ones(len(table_c))
    table_c['z_true'] = np.zeros(len(table_c))

    for i in range(len(table_c)):
        p = np.array([float(s) for s in table_c['PZ_full'][i].split(',')])
        p = p / np.sum(p)
        table_c['z_true'][i] = np.random.choice(z_pdf, p=p)

    table_c = dsigma_table(table_c, 'calibration', z='Z_B', z_true='z_true',
                           w='weight')
    table_c = table_c[table_c['z'] > 0.25 - 1e-6]
    table_c = table_c[table_c['z'] < 1.3 + 1e-6]
    table_c = add_maximum_lens_redshift(table_c, dz_min=0.105)

else:
    raise ValueError("Survey must be 'kids' or 'cfht'.")

# %%

rp_bins = np.logspace(np.log10(0.05), np.log10(15.), 11)
z_bins = [0.15, 0.31, 0.43, 0.54, 0.70]

for lens_bin in range(4):

    table_l = Table.read(os.path.join('raw', '{}_{}_{}_{}.dat'.format(
        args.survey.upper(), 'LOWZ' if lens_bin < 2 else 'CMASS',
        z_bins[lens_bin], z_bins[lens_bin + 1])), delimiter=' ',
                         format='ascii', names=['ra', 'dec', 'z', 'w_sys',
                                                'w_tot', 'm_star'])

    table_r = Table.read(os.path.join('raw', '{}_{}_RANDOMS_{}_{}.dat'.format(
            args.survey.upper(), 'LOWZ' if lens_bin < 2 else 'CMASS',
            z_bins[lens_bin], z_bins[lens_bin + 1])), delimiter=' ',
                         format='ascii', names=['ra', 'dec', 'z'])

    table_r['w_sys'] = np.ones(len(table_r))
    table_l['w_sys'] = table_l['w_tot']

    kwargs = {'n_jobs': multiprocessing.cpu_count(), 'comoving': False,
              'cosmology': FlatLambdaCDM(H0=70, Om0=0.3)}
    if args.survey.lower() == 'cfht':
        kwargs['table_c'] = table_c
    if args.survey.lower() == 'kids':
        kwargs['nz'] = nz

    print('Working on lenses in bin {}...'.format(lens_bin + 1))
    table_l_pre = precompute_catalog(table_l, table_s, rp_bins, **kwargs)
    print('Working on randoms in bin {}...'.format(lens_bin + 1))
    table_r_pre = precompute_catalog(table_r, table_s, rp_bins, **kwargs)

    # Create the jackknife fields.
    table_l_pre = add_continous_fields(table_l_pre, distance_threshold=2)
    centers = jackknife_field_centers(table_l_pre, 100)
    table_l_pre = add_jackknife_fields(table_l_pre, centers)
    table_r_pre = add_jackknife_fields(table_r_pre, centers)

    kwargs = {'return_table': True, 'shear_bias_correction': True,
              'random_subtraction': True,
              'photo_z_dilution_correction': args.survey.lower() == 'cfht',
              'table_r': table_r_pre}

    result = excess_surface_density(table_l_pre, **kwargs)
    kwargs['return_table'] = False
    ds_cov = jackknife_resampling(
        excess_surface_density, table_l_pre, **kwargs)
    result['ds_err'] = np.sqrt(np.diag(ds_cov))

    fname_base = '{}_{}'.format(args.survey.lower(), lens_bin)

    np.savetxt(os.path.join('results', fname_base + '_cov.csv'), ds_cov)

    result.write(os.path.join('results', fname_base + '.csv'),
                 overwrite=True)
