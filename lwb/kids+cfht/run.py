import os
import argparse
import numpy as np
from astropy.table import Table, vstack
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute_catalog, add_maximum_lens_redshift
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from astropy.cosmology import FlatLambdaCDM
from dsigma.surveys import kids

parser = argparse.ArgumentParser(description='Calculate the galaxy-galaxy ' +
                                 'lensing signal for the LWB project.')

parser.add_argument('survey', help='the lens survey')
parser.add_argument('--no_dilution_correction', action='store_true',
                    help='do not correct for photo-z dilution')
parser.add_argument('--nstar', help='whether to calculate only the high or ' +
                    'low n_star regions')
args = parser.parse_args()

if args.nstar is not None:
    if args.nstar not in ['low', 'high']:
        raise ValueError("nstar must be 'low' or 'high'.")
    if args.survey.lower() != 'cfht':
        raise ValueError("nstar can only be specified for CFHT.")

# %%

if args.survey.lower() == 'kids':

    table_s = Table()

    for reg in [9, 12, 15, 23, 'S']:
        kv = Table.read(os.path.join(
            'kids', 'KV450_G{}_reweight_3x4x4_v2_good.cat'.format(reg)), hdu=1)

        table = dsigma_table(kv, 'source', survey='KiDS', version='KV450',
                             verbose=reg == 9)

        for key in table.colnames:
            table[key].unit = None
        table = Table(table)

        table_s = vstack([table_s, table])

    table_s = table_s[table_s['z'] > 0.25 - 1e-6]
    table_s = table_s[table_s['z'] < 1.2 + 1e-6]
    table_s = add_maximum_lens_redshift(table_s, dz_min=0.105)
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'],
                                                  version='KV450')

    table_c = table_s[np.random.randint(len(table_s), size=100000)][
        'z', 'z_l_max', 'w']

    table_c['w_sys'] = np.ones(len(table_c))
    table_c['z_true'] = np.zeros(len(table_c))

    z_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.2]

    for i in range(len(z_bins) - 1):
        nz = Table.read(os.path.join(
            'kids', 'Nz_DIR_z{}t{}.asc'.format(z_bins[i], z_bins[i + 1])),
            format='ascii', names=['z', 'n(z)'])
        nz['z'] = nz['z'] + 0.025
        nz['n(z)'] /= np.sum(nz['n(z)'])
        mask = np.digitize(table_c['z'], z_bins) - 1 == i
        table_c['z_true'][mask] = np.random.choice(
            nz['z'], p=nz['n(z)'], size=np.sum(mask))

elif args.survey.lower() == 'cfht':

    table_s = Table.read('CFHTLS.csv')
    table_c = table_s[np.random.randint(len(table_s), size=100000)]
    table_s = dsigma_table(table_s, 'source', survey='cfhtls')
    table_s = table_s[table_s['z'] > 0.25 - 1e-6]
    table_s = table_s[table_s['z'] < 1.3 + 1e-6]
    table_s = add_maximum_lens_redshift(table_s, dz_min=0.105)

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

if args.no_dilution_correction:
    print("Not correcting for photo-z dilution.")
    table_c['z_l_max'] = np.minimum(table_c['z_l_max'],
                                    table_c['z_true'] - 1e-5)
else:
    print("Correcting for photo-z dilution.")

# %%

rp_bins = np.logspace(np.log10(0.05), np.log10(15.), 11)
z_bins = [0.15, 0.31, 0.43, 0.54, 0.70]

for lens_bin in range(4):

    table_l = Table.read(
        'lwb_files_jan2020/{}_{}_{}_{}.dat'.format(
            args.survey.upper(), 'LOWZ' if lens_bin < 2 else 'CMASS',
            z_bins[lens_bin], z_bins[lens_bin + 1]), delimiter=' ',
        format='ascii', names=['ra', 'dec', 'z', 'w_sys', 'w_tot', 'm_star'])

    table_r = Table.read(
        'lwb_files_jan2020/{}_{}_RANDOMS_{}_{}.dat'.format(
            args.survey.upper(), 'LOWZ' if lens_bin < 2 else 'CMASS',
            z_bins[lens_bin], z_bins[lens_bin + 1]), delimiter=' ',
        format='ascii', names=['ra', 'dec', 'z'])

    if args.nstar is not None:
        mask_l = ((table_l['ra'] < 100) | ((table_l['ra'] > 200) &
                                           (table_l['ra'] < 300)))
        mask_r = ((table_r['ra'] < 100) | ((table_r['ra'] > 200) &
                                           (table_r['ra'] < 300)))
        if args.nstar == 'low':
            table_l = table_l[mask_l]
            table_r = table_r[mask_r]
        else:
            table_l = table_l[~mask_l]
            table_r = table_r[~mask_r]

    table_r['w_sys'] = np.ones(len(table_r))
    table_l['w_sys'] = table_l['w_tot']
    print('Working on lenses in bin {}...'.format(lens_bin + 1))
    table_l_pre = precompute_catalog(table_l, table_s, rp_bins, n_jobs=40,
                                     comoving=False, table_c=table_c,
                                     cosmology=FlatLambdaCDM(H0=70, Om0=0.3))
    print('Working on randoms in bin {}...'.format(lens_bin + 1))
    table_r_pre = precompute_catalog(table_r, table_s, rp_bins, n_jobs=40,
                                     comoving=False, table_c=table_c,
                                     cosmology=FlatLambdaCDM(H0=70, Om0=0.3))

    # Create the jackknife fields.
    table_l_pre = add_continous_fields(table_l_pre, distance_threshold=2)
    centers = jackknife_field_centers(table_l_pre, 100)
    table_l_pre = add_jackknife_fields(table_l_pre, centers)
    table_r_pre = add_jackknife_fields(table_r_pre, centers)

    kwargs = {'return_table': True, 'shear_bias_correction': True,
              'random_subtraction': True, 'photo_z_dilution_correction': True,
              'table_r': table_r_pre}

    result = excess_surface_density(table_l_pre, **kwargs)
    kwargs['return_table'] = False
    ds_cov = jackknife_resampling(
        excess_surface_density, table_l_pre, **kwargs)
    result['ds_err'] = np.sqrt(np.diag(ds_cov))

    fname_base = '{}_{}{}{}'.format(
        args.survey.lower(), lens_bin,
        '_no_dillution_correction' if args.no_dilution_correction else '',
        ('_nstar_' + args.nstar) if args.nstar is not None else '')

    np.savetxt(os.path.join('results', fname_base + '_cov.csv'), ds_cov)

    result.write(os.path.join('results', fname_base + '.csv'),
                 overwrite=True)
