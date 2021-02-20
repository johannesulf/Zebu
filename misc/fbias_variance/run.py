import os
import zebu
import healpy
import argparse
import numpy as np
from functools import partial
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
from astropy import units as u
from dsigma.precompute import precompute_photo_z_dilution_factor
from dsigma.precompute import add_maximum_lens_redshift
from dsigma.stacking import photo_z_dilution_factor

# %%

parser = argparse.ArgumentParser()
parser.add_argument('survey', help='the lensing survey')
args = parser.parse_args()

stage = 1
nside = 42
print('Area: {:.2f} sq. deg'.format(
    healpy.pixelfunc.nside2pixarea(nside, degrees=True)))
n_source_bins = 4 if args.survey != 'kids' else 5
n_lens_bins = 4

# %%


def fbias_func(table_l, table_s):
    return photo_z_dilution_factor(precompute_photo_z_dilution_factor(
        table_l, table_s, cosmology=zebu.cosmo))


if not os.path.isfile('{}.csv'.format(args.survey)):

    fbias_rel_err = np.zeros((n_source_bins, n_lens_bins))

    for source_bin in range(n_source_bins):

        print('Working on source bin {}...'.format(source_bin))

        table_s = zebu.read_raw_data(stage, 'source', source_bin,
                                     survey=args.survey)

        table_s['w_sys'] = 1
        table_s['pix'] = healpy.ang2pix(nside, table_s['ra'], table_s['dec'],
                                        lonlat=True)
        all_pixs = np.unique(table_s['pix'])
        use_pixs = np.zeros(0, dtype=np.int)

        for pix in all_pixs:
            near_pixs = healpy.pixelfunc.get_all_neighbours(nside, pix)
            if np.all(np.isin(near_pixs, all_pixs)):
                use_pixs = np.append(use_pixs, pix)

        table_s = table_s[np.isin(table_s['pix'], use_pixs)]
        table_s = add_maximum_lens_redshift(table_s, dz_min=0.15)

        if 'd_com' not in table_s.colnames:
            table_s['d_com'] = zebu.cosmo.comoving_transverse_distance(
                table_s['z']).to(u.Mpc).value

        if 'd_com_true' not in table_s.colnames:
            table_s['d_com_true'] = zebu.cosmo.comoving_transverse_distance(
                table_s['z_true']).to(u.Mpc).value

        for lens_bin in range(n_lens_bins):

            print('\t...lens bin {}'.format(lens_bin))

            table_l = zebu.read_raw_data(stage, 'lens', lens_bin,
                                         survey=args.survey)
            table_l['z'][:100] = np.percentile(
                table_l['z'], np.arange(100) + 0.5)
            table_l = table_l[:100]

            if np.amin(table_l['z']) < np.amax(table_s['z_l_max']):

                table_s_pix = [table_s[table_s['pix'] == pix] for pix in
                               use_pixs]

                with Pool(cpu_count()) as p:
                    fbias = np.array(p.map(
                        partial(fbias_func, table_l), table_s_pix))

                fbias_rel_err[source_bin, lens_bin] = (
                    np.std(fbias) / np.mean(fbias))

            else:
                fbias_rel_err[source_bin, lens_bin] = np.nan

    np.savetxt('{}.csv'.format(args.survey), fbias_rel_err)

else:
    fbias_rel_err = np.genfromtxt('{}.csv'.format(args.survey))

# %%

plt.imshow(fbias_rel_err, origin='lower', aspect='auto', vmin=0, vmax=0.03)
cb = plt.colorbar()
cb.set_label(r'$\sigma_{f_{\rm bias}} / f_{\rm bias}$')

plt.xticks(-0.5 + np.arange(n_lens_bins + 1), [
    "{:.1f}".format(z) for z in zebu.lens_z_bins])
plt.xlabel('Lens Redshift $z_l$')
plt.yticks(-0.5 + np.arange(n_source_bins + 1), [
    "{:.1f}".format(z) for z in zebu.source_z_bins(stage, survey=args.survey)])
plt.ylabel('Source Redshift $z_s$')

plt.tight_layout(pad=0.3)
plt.savefig('{}.pdf'.format(args.survey))
plt.savefig('{}.png'.format(args.survey), dpi=300)
