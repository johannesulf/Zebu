import os
import numpy as np
import healpy as hp
from tqdm import tqdm
from astropy.table import Table, vstack
from make_mocks import read_buzzard_catalog, ra_dec_in_mock

print('Reading raw buzzard catalog...')
nside = 8
pixel = np.arange(hp.nside2npix(nside))
ra_pixel, dec_pixel = hp.pix2ang(nside, pixel, nest=True, lonlat=True)
pixel_use = pixel[ra_dec_in_mock(ra_pixel, dec_pixel)]
bands = ['g', 'r', 'i', 'z', 'y', 'w1', 'w2']
f_i_cut = 10**((20 - 22.5) / -2.5)

table_b = Table()
for pixel in tqdm(pixel_use):
    table_b_pix = read_buzzard_catalog(pixel)
    f_i = 10**((table_b_pix['mag'][:, bands.index('i')] - 22.5) / -2.5)
    table_b_pix['w'] = f_i / f_i_cut
    table_b_pix = table_b_pix[
        np.random.random(len(table_b_pix)) < table_b_pix['w']]
    table_b_pix['w'] = np.maximum(table_b_pix['w'], 1.0)
    table_b = vstack([table_b, table_b_pix])

table_b.keep_columns(['ra', 'dec', 'w'])
table_b.write(os.path.join('mocks', 'i_band_sample.hdf5'), overwrite=True,
              path='data')
