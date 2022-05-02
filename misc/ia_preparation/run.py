import os
import zebu
import tqdm
import numpy as np
import healpy as hp
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table
from scipy.interpolate import interp1d

# %%

fpath = os.path.join(zebu.base_dir, 'mocks', 'ia.hdf5')

cosmo = FlatLambdaCDM(Om0=0.286, H0=100)

z = np.linspace(0, 4, 1000)
z = interp1d(cosmo.comoving_distance(z), z, kind='cubic')

table_z = Table()
table_z['z_min'] = z(np.arange(80) * 50)
table_z['z_max'] = z((np.arange(80) + 1) * 50)

table_z.write(fpath, path='z')

# %%

nside = 2048
ra, dec = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)

pix_8 = np.genfromtxt(os.path.join(zebu.base_dir, 'misc', 'mock_footprint',
                                   'pixels.txt'), dtype=int)
pix_8_select = np.concatenate(
    [hp.pixelfunc.get_all_neighbours(8, p, nest=True) for p in pix_8])
pix_8_select = np.unique(pix_8_select)
pix_8 = hp.pixelfunc.ang2pix(8, ra, dec, nest=True, lonlat=True)
select = np.isin(pix_8, pix_8_select)

# %%


table_ia = Table()
table_ia['pix'] = np.arange(hp.nside2npix(nside))[select]
table_ia['e_1'] = np.zeros((np.sum(select), 80), dtype=np.float32)
table_ia['e_2'] = np.zeros((np.sum(select), 80), dtype=np.float32)

# %%

directory = '/global/cscratch1/sd/ucapnje/DESI/buzzard4_shear_intrinsic_alignments/'

for i in tqdm.tqdm(range(80)):
    ia = Table.read(directory + 'desi_lensing_shear_buzzard4_{}.fits'.format(
        i))['T'].data.ravel()
    table_ia['e_1'][:, i] = np.real(ia)[select]
    table_ia['e_2'][:, i] = np.imag(ia)[select]

# %%

table_ia.write(fpath, path='ia', append=True, overwrite=True)
