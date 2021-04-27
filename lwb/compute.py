import os
import fitsio
import argparse
import numpy as np
import multiprocessing
from astropy.table import Table, vstack, hstack, join
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute_catalog, add_maximum_lens_redshift
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from dsigma.surveys import des, kids
from astropy.cosmology import FlatLambdaCDM

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal for the LWB project.')

parser.add_argument('survey', help='the lens survey')
args = parser.parse_args()

cosmo = FlatLambdaCDM(100, 0.3)

table_l = vstack([
    Table.read(os.path.join(
        'boss', 'galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz')),
    Table.read(os.path.join(
        'boss', 'galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz'))])
table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC',
                       w_sys=1)
table_l = table_l[table_l['z'] >= 0.15]

if args.survey.lower() == 'des':

    table_s = []

    fname_list = ['mcal-y1a1-combined-riz-unblind-v4-matched.fits',
                  'y1a1-gold-mof-badregion_BPZ.fits',
                  'mcal-y1a1-combined-griz-blind-v3-matched_BPZbase.fits']
    columns_list = [['e1', 'e2', 'R11', 'R12', 'R21', 'R22', 'ra', 'dec',
                     'flags_select', 'flags_select_1p', 'flags_select_1m',
                     'flags_select_2p', 'flags_select_2m'], ['Z_MC'],
                    ['MEAN_Z']]

    for fname, columns in zip(fname_list, columns_list):
        table_s.append(Table(fitsio.read(os.path.join('des', fname),
                                         columns=columns), names=columns))

    table_s = hstack(table_s)
    table_s = dsigma_table(table_s, 'source', survey='DES')
    table_s['z_bin'] = des.tomographic_redshift_bin(table_s['z'])

    for z_bin in range(4):
        use = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[use])
        table_s['R_11'][use] += 0.5 * np.sum(np.diag(R_sel))
        table_s['R_22'][use] += 0.5 * np.sum(np.diag(R_sel))

    table_s = table_s[(table_s['flags_select'] == 0) &
                      (table_s['z_bin'] != -1)]

    table_c = table_s['z', 'z_true', 'w']
    table_c['w_sys'] = 0.5 * (table_s['R_11'] + table_s['R_22'])

    z_s_bins = [0.2, 0.43, 0.63, 0.9, 1.3]
    precompute_kwargs = {'table_c': table_c}
    stacking_kwargs = {'tensor_shear_response_correction': True,
                       'photo_z_dilution_correction': True}

elif args.survey.lower() == 'hsc':

    table_s = Table.read(os.path.join('hsc', 'raw', 'hsc_s16a_lensing.fits'))
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([
        Table.read(os.path.join('hsc', 'raw', 'pdf-s17a_wide-9812.cat.fits')),
        Table.read(os.path.join('hsc', 'raw', 'pdf-s17a_wide-9813.cat.fits'))])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read(os.path.join(
        'hsc', 'raw', 'Afterburner_reweighted_COSMOS_photoz_FDFC.fits'))
    table_c_2.rename_column('S17a_objid', 'id')
    table_c = join(table_c_1, table_c_2, keys='id')
    table_c = dsigma_table(table_c, 'calibration', w_sys='SOM_weight',
                           w='weight_source', z_true='COSMOS_photoz',
                           survey='HSC')

    z_s_bins = np.linspace(0.3, 1.5, 5)
    precompute_kwargs = {'table_c': table_c}
    stacking_kwargs = {'scalar_shear_response_correction': True,
                       'shear_responsivity_correction': True,
                       'photo_z_dilution_correction': True,
                       'hsc_selection_bias_correction': True}

elif args.survey.lower() == 'kids':

    table_s = Table.read(os.path.join(
        'kids', 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits'))
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(table_s['z'],
                                                     version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'], version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_' +
             'v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    nz = np.array([np.genfromtxt(
        os.path.join('kids', fname.format(i + 1))).T for i in range(5)])
    nz[:, 0, :] += 0.025  # in original file, redshifts are lower bin edges

    z_s_bins = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
    precompute_kwargs = {'nz': nz}
    stacking_kwargs = {'scalar_shear_response_correction': True}

else:
    raise ValueError("Survey must be 'des', 'hsc' or 'kids'.")

# %%

table_s = add_maximum_lens_redshift(table_s, dz_min=0.1)
if 'table_c' in precompute_kwargs.keys():
    table_c = add_maximum_lens_redshift(table_c, dz_min=0.1)

precompute_kwargs.update({
    'n_jobs': multiprocessing.cpu_count(), 'comoving': True,
    'cosmology': cosmo})

# %%

rp_bins = np.logspace(np.log10(0.5), np.log10(50.), 11)
z_l_bins = np.linspace(0.2, 0.7, 6)

for lens_bin in range(len(z_l_bins) - 1):

    z_l_min = z_l_bins[lens_bin]
    z_l_max = z_l_bins[lens_bin + 1]

    for source_bin in range(len(z_s_bins) - 1):

        table_l_part = table_l[(z_l_min <= table_l['z']) &
                               (table_l['z'] < z_l_max)]

        z_s_min = z_s_bins[source_bin]
        z_s_max = z_s_bins[source_bin + 1]
        table_s_part = table_s[(z_s_min <= table_s['z']) &
                               (table_s['z'] < z_s_max)]
        if 'table_c' in precompute_kwargs.keys():
            table_c_part = table_c[(z_s_min <= table_c['z']) &
                                   (table_c['z'] < z_s_max)]
            precompute_kwargs['table_c'] = table_c_part
            table_l_part = table_l_part[table_l_part['z'] < np.amax(
                table_c_part['z_l_max'])]

        if np.amin(table_l_part['z']) >= np.amax(table_s_part['z_l_max']):
            continue

        table_l_pre = precompute_catalog(table_l_part, table_s_part, rp_bins,
                                         **precompute_kwargs)

        # Create the jackknife fields.
        table_l_pre = add_continous_fields(table_l_pre, distance_threshold=2)
        centers = jackknife_field_centers(table_l_pre, 100)
        table_l_pre = add_jackknife_fields(table_l_pre, centers)

        stacking_kwargs['return_table'] = True
        result = excess_surface_density(table_l_pre, **stacking_kwargs)
        stacking_kwargs['return_table'] = False
        ds_cov = jackknife_resampling(
            excess_surface_density, table_l_pre, **stacking_kwargs)
        result['ds_err'] = np.sqrt(np.diag(ds_cov))

        fname_base = '{}_l{}_s{}'.format(args.survey.lower(), lens_bin,
                                         source_bin)

        np.savetxt(os.path.join('results', fname_base + '_cov.csv'), ds_cov)

        result.write(os.path.join('results', fname_base + '.csv'),
                     overwrite=True)
