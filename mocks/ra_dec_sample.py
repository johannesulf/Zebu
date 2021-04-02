import os
import numpy as np
import healpy as hp
from astropy.table import Table, vstack
from make_mocks import read_buzzard_catalog, ra_dec_in_region

region = 1
nside = 8
pixel = np.arange(hp.nside2npix(nside))
ra_pixel, dec_pixel = hp.pix2ang(nside, pixel, nest=True, lonlat=True)
pixel_use = pixel[ra_dec_in_region(ra_pixel, dec_pixel, region)]

table_b = Table()
for pixel in pixel_use:
    table_b = vstack([table_b, read_buzzard_catalog(pixel)[::10]])
table_b = table_b.filled()
table_b.meta['bands'] = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
table_b.keep_columns(['ra', 'dec'])
table_b.write(os.path.join('region_{}'.format(region), 'ra_dec_sample.hdf5'),
              overwrite=True, path='data')
