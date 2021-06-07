import zebu
import dsigma
import numpy as np
import matplotlib.pyplot as plt
import treecorr

# %%

survey = 'DES'
lens_bin = 2

fig, (ax1, ax2) = plt.subplots(figsize=(7, 3), ncols=2, sharey=True)

# %%

table_l = zebu.read_mock_data('lens', lens_bin)
table_r = zebu.read_mock_data('random', lens_bin)
table_s = zebu.read_mock_data('source', 'all', survey=survey)
table_s['m'] = np.where(
    (zebu.lens_z_bins[lens_bin] - 0.05 < table_s['z_true']) &
    (table_s['z_true'] < zebu.lens_z_bins[lens_bin + 1] + 0.05), 0, 1)

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])
dz_min_list = np.linspace(0, 0.4, 5)
color_list = plt.get_cmap('viridis')(np.linspace(0.0, 0.9, len(dz_min_list)))

for color, dz_min in zip(color_list, dz_min_list):

    print(dz_min)

    dsigma.precompute.add_maximum_lens_redshift(table_s, dz_min=dz_min)
    dsigma.precompute.add_precompute_results(
        table_l, table_s, zebu.rp_bins, cosmology=zebu.cosmo, n_jobs=4)
    dsigma.precompute.add_precompute_results(
        table_r, table_s, np.array([0, 5]), cosmology=zebu.cosmo, n_jobs=4)

    b = (dsigma.stacking.scalar_shear_response_factor(table_r)[0] /
         dsigma.stacking.scalar_shear_response_factor(table_l))

    ax1.plot(rp, b, color=color,
             label=r'$\Delta z = {:.1f}$'.format(dz_min))

ax1.axhline(1.0, color='black', ls='--')
ax1.set_title(r'$\Delta\Sigma$')
ax1.legend(loc='best')
ax1.set_xscale('log')
ax1.set_xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
ax1.set_ylabel(r'Boost $b$')

# %%

cat_l = treecorr.Catalog(ra=table_l['ra'], dec=table_l['dec'],
                         ra_units='deg', dec_units='deg')
cat_r = treecorr.Catalog(ra=table_r['ra'], dec=table_r['dec'],
                         ra_units='deg', dec_units='deg')

theta = np.sqrt(zebu.theta_bins[1:] * zebu.theta_bins[:-1])
source_bin_list = np.arange(len(zebu.source_z_bins[survey.lower()]) - 1)
color_list = plt.get_cmap('viridis')(
    np.linspace(0.0, 0.9, len(source_bin_list)))

for color, source_bin in zip(color_list, source_bin_list):

    print(source_bin)

    table_s = zebu.read_mock_data('source', source_bin, survey=survey)

    table_s['m'] = np.where(
        (zebu.lens_z_bins[lens_bin] - 0.05 < table_s['z_true']) &
        (table_s['z_true'] < zebu.lens_z_bins[lens_bin + 1] + 0.05), 0, 1)

    cat_s = treecorr.Catalog(ra=table_s['ra'], dec=table_s['dec'],
                             ra_units='deg', dec_units='deg',
                             k=table_s['m'])

    nk = treecorr.NKCorrelation(
        max_sep=np.amax(zebu.theta_bins), min_sep=np.amin(zebu.theta_bins),
        nbins=len(zebu.theta_bins) - 1, sep_units='arcmin', metric='Arc',
        bin_slop=0)
    nk.process(cat_l, cat_s)

    b = np.mean(table_s['m']) / nk.xi

    ax2.plot(theta, b, color=color,
             label=r'${:.2f} \leq z_s < {:.2f}$'.format(
                 zebu.source_z_bins[survey.lower()][source_bin],
                 zebu.source_z_bins[survey.lower()][source_bin + 1]))

ax2.axhline(1.0, color='black', ls='--')
ax2.set_title(r'$\gamma_t$')
ax2.legend(loc='best')
ax2.set_xscale('log')
ax2.set_xlabel(r'Angle $\theta \, [\mathrm{arcmin}]$')

# %%

plt.tight_layout(pad=0.3)
plt.subplots_adjust(wspace=0)
plt.savefig('boost_vs_cut.pdf')
plt.savefig('boost_vs_cut.png', dpi=300)
