import argparse
import numpy as np
import multiprocessing
from pathlib import Path
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, vstack, join

from dsigma.surveys import des, kids
from dsigma.helpers import dsigma_table
from dsigma.precompute import precompute
from dsigma.stacking import excess_surface_density

parser = argparse.ArgumentParser(
    description='Calculate the lensing signal for a LWB-type analysis.')

parser.add_argument('survey', help='the lens survey',
                    choices=['des', 'hsc', 'kids'])
args = parser.parse_args()

cosmology = FlatLambdaCDM(100, 0.3)

path = Path('boss')
table_l = vstack([
    Table.read(path / 'galaxy_DR12v5_CMASSLOWZTOT_South.fits.gz'),
    Table.read(path / 'galaxy_DR12v5_CMASSLOWZTOT_North.fits.gz')])

table_l['w_sys'] = table_l['WEIGHT_SYSTOT'] * (
    table_l['WEIGHT_NOZ'] + table_l['WEIGHT_CP'] - 1)
table_l = dsigma_table(table_l, 'lens', z='Z', ra='RA', dec='DEC')
table_l = table_l[table_l['z'] >= 0.15]

if args.survey.lower() == 'des':

    path = Path('des')
    table_s = Table.read(path / 'des_y3.hdf5', path='catalog')
    table_s = dsigma_table(table_s, 'source', survey='DES')

    for z_bin in range(4):
        select = table_s['z_bin'] == z_bin
        R_sel = des.selection_response(table_s[select])
        print("Bin {}: R_sel = {:.1f}%".format(
            z_bin + 1, 100 * 0.5 * np.sum(np.diag(R_sel))))
        table_s['R_11'][select] += 0.5 * np.sum(np.diag(R_sel))
        table_s['R_22'][select] += 0.5 * np.sum(np.diag(R_sel))

    table_s = table_s[table_s['flags_select']]
    table_s['m'] = des.multiplicative_shear_bias(
        table_s['z_bin'], version='Y3')

    table_n = Table.read(path / 'des_y3.hdf5', path='redshift')
    z_mean = np.array([np.average(table_n['z'], weights=table_n['n'][:, i])
                       for i in range(4)])
    table_s['z'] = z_mean[table_s['z_bin']]

    z_s_bins = np.array([0.0, 0.358, 0.631, 0.872, 2.0])
    precompute_kwargs = dict(table_n=table_n, lens_source_cut=None)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           matrix_shear_response_correction=True)

elif args.survey.lower() == 'hsc':

    path = Path('hsc')
    table_s = Table.read(path / 'hsc_s16a_lensing.fits')
    table_s = dsigma_table(table_s, 'source', survey='HSC')

    table_c_1 = vstack([Table.read(path / 'pdf-s17a_wide-9812.cat.fits'),
                        Table.read(path / 'pdf-s17a_wide-9813.cat.fits')])
    for key in table_c_1.colnames:
        table_c_1.rename_column(key, key.lower())
    table_c_2 = Table.read(
        path / 'Afterburner_reweighted_COSMOS_photoz_FDFC.fits')
    table_c_2.rename_column('S17a_objid', 'id')
    table_c = join(table_c_1, table_c_2, keys='id')
    table_c = dsigma_table(table_c, 'calibration', w_sys='SOM_weight',
                           w='weight_source', z_true='COSMOS_photoz',
                           survey='HSC')

    z_s_bins = np.linspace(0.3, 1.5, 5)
    precompute_kwargs = dict(table_c=table_c, lens_source_cut=None)
    stacking_kwargs = dict(scalar_shear_response_correction=True,
                           shear_responsivity_correction=True,
                           photo_z_dilution_correction=True,
                           hsc_selection_bias_correction=True)

elif args.survey.lower() == 'kids':

    path = Path('kids')
    table_s = Table.read(path / 'KiDS_DR4.1_ugriZYJHKs_SOM_gold_WL_cat.fits')
    table_s = dsigma_table(table_s, 'source', survey='KiDS')

    table_s['z_bin'] = kids.tomographic_redshift_bin(table_s['z'],
                                                     version='DR4')
    table_s['m'] = kids.multiplicative_shear_bias(table_s['z_bin'],
                                                  version='DR4')
    table_s = table_s[table_s['z_bin'] >= 0]
    table_s['z'] = np.array([0.1, 0.3, 0.5, 0.7, 0.9])[table_s['z_bin']]

    fname = ('K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_' +
             'v2_SOMcols_Fid_blindC_TOMO{}_Nz.asc')
    table_n = Table()
    table_n['z'] = np.genfromtxt(path / fname.format(1))[:, 0] + 0.025
    table_n['n'] = np.vstack([np.genfromtxt(
        path / fname.format(i + 1))[:, 1] for i in range(5)]).T

    z_s_bins = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.2])
    precompute_kwargs = dict(table_n=table_n, lens_source_cut=None)
    stacking_kwargs = dict(scalar_shear_response_correction=True)


precompute_kwargs.update(dict(
    comoving=True, cosmology=cosmology, progress_bar=True,
    n_jobs=multiprocessing.cpu_count()))
stacking_kwargs.update(dict(return_table=True))

path = Path('results')
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

        if args.survey == 'hsc':
            table_c_part = table_c[(z_s_min <= table_c['z']) &
                                   (table_c['z'] < z_s_max)]
            precompute_kwargs['table_c'] = table_c_part

            # Don't compute cases where f_bias would be undefined.
            if np.amax(table_l_part['z']) >= np.amax(table_c_part['z']):
                continue

        precompute(table_l_part, table_s_part, rp_bins, **precompute_kwargs)
        result = excess_surface_density(table_l_part, **stacking_kwargs)

        fname = '{}_l{}_s{}.csv'.format(
            args.survey.lower(), lens_bin, source_bin)
        result.write(path / fname, overwrite=True)
