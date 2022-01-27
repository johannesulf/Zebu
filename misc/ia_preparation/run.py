import zebu
import numpy as np
import healpy as hp
from astropy.table import Table
from scipy.interpolate import interp1d

# %%

data = np.load('desi_lensing_ia_nla_0p49_0.npy')

z = np.linspace(0, 4, 1000)
z = interp1d(zebu.cosmo.comoving_distance(z), z, kind='cubic')

table_z = Table()
table_z['z_min'] = z(np.arange(data.shape[0]) * 50)
table_z['z_max'] = z((np.arange(data.shape[0]) + 1) * 50)

table_z.write('ia.hdf5', path='z')


def ra_dec_in_mock(ra, dec):

    nside = 8
    pix = hp.ang2pix(nside, np.array(ra), np.array(dec), nest=True,
                     lonlat=True)

    pix_use = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
               23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37, 38, 39,
               48, 49, 50, 51, 52, 53, 54, 55, 69, 70, 71, 73, 74, 75, 76, 77,
               78, 79, 80, 82, 83, 88, 89, 90, 91, 96, 97, 98, 99, 100, 101,
               102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
               115, 120, 121, 122, 123, 373, 374, 375, 377, 378, 379, 380, 381,
               382, 383]

    return np.isin(pix, pix_use)


ra, dec = hp.pix2ang(
    hp.npix2nside(data.shape[1]), np.arange(data.shape[1]), lonlat=True)
use = ra_dec_in_mock(ra, dec)

table_ia = Table()
table_ia['pix'] = np.arange(data.shape[1])[use]
table_ia['e_1'] = np.real(data[:, use]).astype(np.float32).T
table_ia['e_2'] = np.imag(data[:, use]).astype(np.float32).T

table_ia.write('ia.hdf5', path='ia', append=True, overwrite=True)
