import numpy as np
import healpy as hp
import skymapper as skm
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from scipy.spatial.transform import Rotation


def mean_of_coordinates(ra, dec):

    vec = hp.ang2vec(ra, dec, lonlat=True)
    ra_center, dec_center = hp.vec2ang(np.mean(vec, axis=0), lonlat=True)

    return ra_center[0], dec_center[0]


nside = 8

fstream = open('buzzard_files.txt')
pix = []

for line in fstream.readlines():
    pix.append(int(line.split('.')[-2]))

fstream.close()

in_buzzard = np.array([i in pix for i in range(hp.nside2npix(nside))])
ra_p, dec_p, vertices = skm.healpix.getGrid(nside, return_vertices=True,
                                            nest=True)[1:4]

pix_inner = []
for p in pix:
    neighbors = hp.get_all_neighbours(nside, p, nest=True)
    neighbors = neighbors[neighbors != -1]
    neighbors = hp.get_all_neighbours(nside, neighbors, nest=True).flatten()
    if np.all([n in pix or n == -1 for n in neighbors]):
        pix_inner.append(p)

in_inner_buzzard = np.array(
    [i in pix_inner for i in range(hp.nside2npix(nside))])

table_t_bright = Table.read('bright_tiles_ngc.fits')
table_t_dark = Table.read('dark_tiles_ngc.fits')
ra_t, dec_t = mean_of_coordinates(table_t_bright['RA'], table_t_bright['DEC'])
c_t = SkyCoord(ra_t, dec_t, unit='deg')
vec_t = hp.ang2vec(ra_t, dec_t, lonlat=True)

ra_b, dec_b = mean_of_coordinates(ra_p[in_buzzard],
                                  dec_p[in_buzzard])
c_b = SkyCoord(ra_b, dec_b, unit='deg')
vec_b = hp.ang2vec(ra_b, dec_b, lonlat=True)

# apply a rotation such that centers of tiles and buzzard overlap
rotation_axis = np.cross(vec_t, vec_b)
rotation_axis /= np.sqrt(np.sum(rotation_axis**2))
rotation_angle = c_b.separation(c_t).to(u.rad).value
rotmat = Rotation.from_rotvec(
    rotation_angle * rotation_axis).as_matrix()
for table_t in [table_t_bright, table_t_dark]:
    table_t['RA'], table_t['DEC'] = hp.rotator.rotateDirection(
        rotmat, table_t['RA'], table_t['DEC'], lonlat=True)
    table_t['RA'] = np.where(table_t['RA'] < 0, table_t['RA'] + 360,
                             table_t['RA'])

# rotate tiles additionally by hand to maximize overlap
rotmat = Rotation.from_rotvec(vec_b * 1.6).as_matrix()
for table_t in [table_t_bright, table_t_dark]:
    table_t['RA'], table_t['DEC'] = hp.rotator.rotateDirection(
        rotmat, table_t['RA'], table_t['DEC'], lonlat=True)
    table_t['RA'] = np.where(table_t['RA'] < 0, table_t['RA'] + 360,
                             table_t['RA'])

hmap = hp.ud_grade(in_inner_buzzard, 512,
                   order_in='NESTED', order_out='NESTED')
ra, dec = hp.pix2ang(512, np.arange(len(hmap))[hmap], nest=True,
                     lonlat=True)
c_p = SkyCoord(ra, dec, unit='deg')

c_t = SkyCoord(table_t_bright['RA'], table_t_bright['DEC'], unit='deg')
sep2d = c_t.match_to_catalog_sky(c_p)[1]
table_t_bright = table_t_bright[sep2d < 2 * u.deg]

c_t = SkyCoord(table_t_dark['RA'], table_t_dark['DEC'], unit='deg')
sep2d = c_t.match_to_catalog_sky(c_p)[1]
table_t_dark = table_t_dark[sep2d < 2 * u.deg]

fig = plt.figure(figsize=(7.0, 3.5))
crit = skm.stdDistortion
proj = skm.Albers.optimize(ra_p[in_buzzard], dec_p[in_buzzard], crit=crit)
footprint = skm.Map(proj, facecolor='white', ax=fig.gca())
footprint.vertex(vertices[in_buzzard], facecolor='grey')
footprint.vertex(vertices[in_inner_buzzard], facecolor='darkgrey')
footprint.scatter(table_t_dark['RA'], table_t_dark['DEC'], s=0.1)
footprint.focus(ra_p[in_buzzard], dec_p[in_buzzard])
footprint.savefig('footprint.pdf')
footprint.savefig('footprint.png', dpi=300)

np.savetxt('pixels.txt', np.arange(len(in_inner_buzzard))[in_inner_buzzard],
           fmt='%d')
table_t_bright['PROGRAM'] = 'DARK'
table_t_dark['PROGRAM'] = 'DARK'
table_t_bright.write('bright_tiles_mock.fits', overwrite=True)
table_t_dark.write('dark_tiles_mock.fits', overwrite=True)
