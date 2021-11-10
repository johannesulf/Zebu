import zebu
import healpy
import numpy as np
from astropy.table import vstack
from dsigma.precompute import photo_z_dilution_factor
from dsigma.precompute import add_maximum_lens_redshift
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# %%

for survey, color in zip(['DES', 'HSC', 'KiDS'],
                         ['red', 'royalblue', 'purple']):

    table_c = vstack([zebu.read_mock_data('calibration', i, survey=survey)
                      for i in range(4 if survey != 'KiDS' else 5)])[::3]

    table_c_plus = table_c.copy()
    table_c_minus = table_c.copy()
    table_c_plus['z_true'] += 0.01
    table_c_minus['z_true'] -= 0.01

    z_l_all = np.linspace(0.1, 0.9, 50)
    dz_min_all = np.linspace(0.0, 0.4, 30)
    d_z_s = np.zeros((len(z_l_all), len(dz_min_all)))

    for i, z_l in enumerate(z_l_all):
        for k, dz_min in enumerate(dz_min_all):

            add_maximum_lens_redshift(table_c, dz_min=dz_min)
            add_maximum_lens_redshift(table_c_plus, dz_min=dz_min)
            add_maximum_lens_redshift(table_c_minus, dz_min=dz_min)

            f_bias = photo_z_dilution_factor(z_l, table_c, zebu.cosmo)
            f_bias_plus = photo_z_dilution_factor(z_l, table_c_plus,
                                                  zebu.cosmo)
            f_bias_minus = photo_z_dilution_factor(z_l, table_c_minus,
                                                   zebu.cosmo)

            d_f_bias_d_z_s = (f_bias_plus - f_bias_minus) / 0.02 / f_bias

            d_z_s[i, k] = 0.01 / np.abs(d_f_bias_d_z_s)

    cs = plt.contour(dz_min_all, z_l_all, d_z_s * 100, levels=[0.4, 0.7, 1.0],
                     colors=color, linestyles=['-', '--', ':'])

    if survey.lower() == 'des':
        plt.clabel(
            cs, fmt=r'$\sigma_{\langle z_s \rangle} = %.1f \times 10^{-2}$')

for offset, survey, color in zip([0, 1, 2], ['DES', 'HSC', 'KiDS'],
                                 ['red', 'royalblue', 'purple']):
    plt.text(0.05, 0.95 - 0.1 * offset, survey, horizontalalignment='left',
             verticalalignment='top', transform=plt.gca().transAxes,
             color=color)

plt.title(r'Requirement for $1\%$ Calibration')
plt.xlabel(r'Lens-Source Separation $\Delta z_{\rm min}$')
plt.ylabel(r'Lens Redshift $z_l$')
plt.tight_layout(pad=0.3)
plt.savefig('mean_redshift_requirement.pdf')
plt.savefig('mean_redshift_requirement.png', dpi=300)
plt.close()

# %%

nside = 59
print('Area: {:.2f} sq. deg'.format(
    healpy.pixelfunc.nside2pixarea(nside, degrees=True)))

for survey, color in zip(['DES', 'HSC', 'KiDS'],
                         ['red', 'royalblue', 'purple']):
    table_s = vstack([zebu.read_mock_data('source', i, survey=survey)
                      for i in range(4 if survey != 'KiDS' else 5)])
    table_s['pix'] = healpy.ang2pix(nside, table_s['ra'], table_s['dec'],
                                    lonlat=True)

    # remove incomplete pixels close to the mock edge
    all_pixs = np.unique(table_s['pix'])
    use_pixs = np.zeros(0, dtype=int)

    for pix in all_pixs:
        near_pixs = healpy.pixelfunc.get_all_neighbours(nside, pix)
        if np.all(np.isin(near_pixs, all_pixs)):
            use_pixs = np.append(use_pixs, pix)

    table_s = table_s[np.isin(table_s['pix'], use_pixs)]

    # reweight pencil beams such that they reproduce photo-z distribution
    table_s['z_dig'] = np.digitize(
        table_s['z'], np.linspace(np.amin(zebu.source_z_bins[survey.lower()]),
                                  np.amax(zebu.source_z_bins[survey.lower()]),
                                  31)) - 1
    table_s['w_sys'] = 1.0

    for pix in np.unique(table_s['pix']):
        use = table_s['pix'] == pix
        table_s['w_sys'][use] = (
            np.bincount(table_s['z_dig'], minlength=30) / float(len(use)) /
            np.bincount(table_s['z_dig'][use], minlength=30) *
            float(np.sum(use)))[table_s['z_dig'][use]]

    z_l_all = np.linspace(0.3, 0.7, 3)
    dz_min_all = np.linspace(0.0, 0.4, 10)
    f_bias_acc = np.zeros((len(z_l_all), len(dz_min_all)))

    for i, z_l in enumerate(z_l_all):
        print(z_l)
        for k, dz_min in enumerate(dz_min_all):

            add_maximum_lens_redshift(table_s, dz_min=dz_min)

            f_bias_all = photo_z_dilution_factor(z_l, table_s, zebu.cosmo)
            f_bias = []

            for pix in np.unique(table_s['pix'])[:50]:
                use = (table_s['pix'] == pix) & (table_s['z_l_max'] > z_l)
                f_bias.append(photo_z_dilution_factor(z_l, table_s[use],
                                                      zebu.cosmo))

            f_bias_acc[i, k] = np.std(f_bias) / f_bias_all

    for i, ls in enumerate(['-', '--', ':']):
        f_bias_acc_interp = interp1d(dz_min_all, f_bias_acc[i], kind='cubic')
        dz_min_plot = np.linspace(0.0, 0.4, 100)
        plt.plot(dz_min_plot, f_bias_acc_interp(dz_min_plot), ls=ls,
                 label=r'$z_l = {:.1f}$'.format(z_l_all[i]) if survey == 'DES'
                 else None, color=color)

plt.legend(loc='best')
plt.xlabel(r'Lens-Source Separation $\Delta z_{\rm min}$')
plt.ylabel(r'Calibration Accuracy $\sigma_{f_{\rm bias}} / f_{\rm bias}$')
plt.tight_layout(pad=0.3)
plt.xlim(0.0, 0.4)
plt.savefig('f_bias_accuracy_1sqdeg.pdf')
plt.savefig('f_bias_accuracy_1sqdeg.png', dpi=300)
