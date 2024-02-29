import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import zebu

from astropy.io.ascii import convert_numpy
from astropy.table import Table, vstack
from dsigma.stacking import boost_factor

# %%

converters = {'*': [convert_numpy(typ) for typ in (int, float, bool, str)]}
table_c = Table.read(zebu.BASE_PATH / 'stacks' / 'config.csv',
                     converters=converters)

# %%

for sources in ['des', 'kids']:
    for i in range(5):
        lens_bin = i % 3
        lenses = 'bgs' if i < 3 else 'lrg'
        n_s = 4 if sources != 'kids' else 5

        config = table_c['configuration'][
            (table_c['lenses'] == lenses) & (table_c['sources'] == sources) &
            (table_c['photometric redshifts'])][0]

        table_l = vstack([Table.read(
            zebu.BASE_PATH / 'stacks' / 'results' / '{}'.format(
                config) / 'l{}_s{}_ds.hdf5'.format(lens_bin, source_bin)) for
            source_bin in range(n_s)])

        config = table_c['configuration'][
            (table_c['lenses'] == lenses + '-r') &
            (table_c['sources'] == sources) &
            (table_c['photometric redshifts'])][0]

        table_r = vstack([Table.read(
            zebu.BASE_PATH / 'stacks' / 'results' / '{}'.format(
                config) / 'l{}_s{}_ds.hdf5'.format(lens_bin, source_bin)) for
            source_bin in range(n_s)])

        b = boost_factor(table_l, table_r)

        rp = np.sqrt(zebu.RP_BINS[1:] * zebu.RP_BINS[:-1])
        color = mpl.colormaps['plasma'](i / 5.0)
        plt.plot(rp, b - 1, color=color, label='{}-{}'.format(
            lenses.upper(), lens_bin + 1))

    plt.axvspan(0, 1, color='lightgrey', zorder=-99)
    plt.xscale('log')
    plt.legend(loc='best', frameon=False)
    plt.xlabel(r'Projected radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
    plt.ylabel(r'Angular clustering $w_i$')
    plt.tight_layout(pad=0.8)
    plt.savefig('clustering_{}.pdf'.format(sources))
    plt.savefig('clustering_{}.png'.format(sources), dpi=300)
    plt.close()
