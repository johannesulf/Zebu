import zebu
import healpy
import numpy as np
from dsigma.precompute import photo_z_dilution_factor
from dsigma.precompute import add_maximum_lens_redshift
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# %%

for survey, color in zip(['DES', 'HSC', 'KiDS'],
                         ['red', 'royalblue', 'purple']):

    table_c = zebu.read_mock_data('calibration', 'all', survey=survey)

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

            if z_l > np.amax(table_c['z_l_max']):
                continue

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
plt.xlabel(r'Lens-source separation $\Delta z_{\rm min}$')
plt.ylabel(r'Lens sedshift $z_l$')
plt.tight_layout(pad=0.3)
plt.savefig('mean_redshift_requirement.pdf')
plt.savefig('mean_redshift_requirement.png', dpi=300)
plt.close()

# %%

nside = 42
print('Area: {:.2f} sq. deg'.format(
    healpy.pixelfunc.nside2pixarea(nside, degrees=True)))

for dz_min in [0.0, 0.2, 0.4]:
    for survey, color in zip(['DES', 'HSC', 'KiDS'],
                             ['red', 'royalblue', 'purple']):
        table_s = zebu.read_mock_data('source', 'all', survey=survey)
        table_s['w_sys'] = 1.0
        add_maximum_lens_redshift(table_s, dz_min=dz_min)
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

        z_l_all = np.linspace(0.1, 0.9, 9)
        f_bias_acc = np.zeros(len(z_l_all))

        for i, z_l in enumerate(z_l_all):

            if z_l > np.amax(table_s['z_l_max']):
                continue

            f_bias_all = photo_z_dilution_factor(z_l, table_s, zebu.cosmo)
            f_bias = []

            for pix in np.unique(table_s['pix'])[:30]:
                use = (table_s['pix'] == pix) & (table_s['z_l_max'] > z_l)
                f_bias.append(photo_z_dilution_factor(z_l, table_s[use],
                                                      zebu.cosmo))

            f_bias_acc[i] = np.std(f_bias) / f_bias_all

        use = f_bias_acc > 0
        f_bias_acc_interp = interp1d(z_l_all[use], f_bias_acc[use],
                                     kind='cubic', bounds_error=False)
        z_l_plot = np.linspace(np.amin(z_l_all), np.amax(z_l_all), 100)
        plt.plot(z_l_plot, f_bias_acc_interp(z_l_plot), label=survey,
                 color=color)

    plt.legend(loc='best')
    plt.xlabel(r'Lens redshift $z_l$')
    plt.ylabel(r'Calibration precision $\sigma_{f_{\rm bias}} / f_{\rm bias}$')
    plt.tight_layout(pad=0.3)
    plt.xlim(np.amin(z_l_all), np.amax(z_l_all))
    plt.savefig('f_bias_accuracy_2sqdeg_dzmin{}.pdf'.format(
        str(dz_min).replace('.', 'p')))
    plt.savefig('f_bias_accuracy_2sqdeg_dzmin{}.png'.format(
        str(dz_min).replace('.', 'p')), dpi=300)
    plt.close()
