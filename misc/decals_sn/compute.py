import os
import zebu
import argparse
import numpy as np
from astropy.table import Table, vstack, hstack, join
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute_catalog, add_maximum_lens_redshift
from dsigma.jackknife import add_continous_fields, jackknife_field_centers
from dsigma.jackknife import add_jackknife_fields, jackknife_resampling
from dsigma.stacking import excess_surface_density
from dsigma.surveys import kids
from astropy.cosmology import FlatLambdaCDM

# %%

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal for the DECaLS.')

parser.add_argument('survey', help='the lens survey')
args = parser.parse_args()

table_l = Table()

for part in ['north', 'south']:
    table_l = vstack([table_l, hstack([
        Table.read('dr9_sv3_lrg_{}_0.57.0_basic.fits'.format(part)),
        Table.read('dr9_sv3_lrg_{}_0.57.0_pz.fits'.format(part))])])

# %%

table_l = dsigma_table(table_l, 'lens', ra='RA', dec='DEC', z='Z_PHOT_MEAN',
                       w_sys=1)

# %%

if args.survey.lower() == 'hsc':

    table_s = Table.read(os.path.join(
        zebu.base_dir, 'lwb', 'hsc', 'raw', 'hsc_s16a_lensing.fits'))
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([
        Table.read(os.path.join(zebu.base_dir, 'lwb', 'hsc', 'raw',
                                'pdf-s17a_wide-9812.cat.fits')),
        Table.read(os.path.join(zebu.base_dir, 'lwb', 'hsc', 'raw',
                                'pdf-s17a_wide-9813.cat.fits'))])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read(os.path.join(
        zebu.base_dir, 'lwb', 'hsc', 'raw',
        'Afterburner_reweighted_COSMOS_photoz_FDFC.fits'))
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
        zebu.base_dir, 'lwb', 'kids', 'raw',
        'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits'))
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(table_s['z'],
                                                     version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z'], version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_' +
             'v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    nz = np.array([np.genfromtxt(
        os.path.join(zebu.base_dir, 'lwb', 'kids', 'raw',
                     fname.format(i + 1))).T for i in range(5)])
    nz[:, 0, :] += 0.025  # in original file, redshifts are lower bin edges

    z_s_bins = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
    precompute_kwargs = {'nz': nz}
    stacking_kwargs = {'scalar_shear_response_correction': True}

else:
    raise ValueError("Survey must be 'hsc' or 'kids'.")

# %%

table_s = add_maximum_lens_redshift(table_s, dz_min=0.1)
if 'table_c' in precompute_kwargs.keys():
    table_c = add_maximum_lens_redshift(table_c, dz_min=0.1)

precompute_kwargs.update({'n_jobs': 4, 'comoving': True,
                          'cosmology': FlatLambdaCDM(100, 0.3)})

# %%

rp_bins = np.logspace(np.log10(0.5), np.log10(25), 10)
z_l_bins = [0.1, 0.3, 0.5]

for lens_bin in range(len(z_l_bins) - 1):

    z_l_min = z_l_bins[lens_bin]
    z_l_max = z_l_bins[lens_bin + 1]

    table_l_part = table_l[(z_l_min <= table_l['z']) &
                           (table_l['z'] < z_l_max)]

    table_l_pre = precompute_catalog(table_l_part, table_s, rp_bins,
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

    fname_base = '{:.1f}_{:.1f}_{}'.format(
        z_l_min, z_l_max, args.survey.lower()).replace('.', 'p')

    result.write(fname_base + '.csv', overwrite=True)
    np.savetxt(fname_base + '_cov.csv', ds_cov)
