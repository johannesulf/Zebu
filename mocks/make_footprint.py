import os
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


def main():

    fpath = os.path.join('/', 'project', 'projectdirs', 'desi', 'mocks',
                         'buzzard', 'buzzard_v2.0', 'buzzard-4',
                         'addgalspostprocess', 'truth')
    fname_list = os.listdir(fpath)
    pixel = []
    for fname in fname_list:
        pixel.append(int(fname.split('.')[-2]))

    nside = 8
    in_buzzard = np.isin(range(hp.nside2npix(nside)), pixel)
    ra_p, dec_p, vertices = skm.healpix.getGrid(nside, return_vertices=True,
                                                nest=True)[1:4]

    pixel_inner = []
    for p in pixel:
        neighbors = hp.get_all_neighbours(nside, p, nest=True)
        neighbors = neighbors[neighbors != -1]
        neighbors = hp.get_all_neighbours(
            nside, neighbors, nest=True).flatten()
        neighbors = neighbors[neighbors != -1]
        if np.all(np.isin(neighbors, pixel)):
            pixel_inner.append(p)

    in_inner_buzzard = np.isin(range(hp.nside2npix(nside)), pixel_inner)
    np.savetxt(os.path.join('buzzard-4', 'pixels.csv'), pixel_inner, fmt='%i')

    fpath = os.path.join('/', 'global', 'cfs', 'cdirs', 'desi', 'survey',
                         'ops', 'surveyops', 'trunk', 'ops')
    table_t = Table.read(os.path.join(fpath, 'tiles-main.ecsv'))
    c = SkyCoord(table_t['RA'], table_t['DEC'], unit='deg')
    # Cut the tiles to just the NGC region.
    table_t = table_t[c.separation(SkyCoord(180, 40, unit='deg')) < 80 * u.deg]
    table_t['RUNDATE'] = '2021-04-06T00:39:37'
    table_t['FIELDROT'] = 1.0
    table_t['FA_HA'] = 0.0
    table_t['OBSCONDITIONS'] = 15
    table_t['IN_DESI'] = 1

    ra_t, dec_t = mean_of_coordinates(table_t['RA'], table_t['DEC'])
    c_t = SkyCoord(ra_t, dec_t, unit='deg')
    vec_t = hp.ang2vec(ra_t, dec_t, lonlat=True)

    ra_b, dec_b = mean_of_coordinates(ra_p[in_buzzard],
                                      dec_p[in_buzzard])
    c_b = SkyCoord(ra_b, dec_b, unit='deg')
    vec_b = hp.ang2vec(ra_b, dec_b, lonlat=True)

    # Apply a rotation such that centers of tiles and buzzard pixels overlap.
    rotation_axis = np.cross(vec_t, vec_b)
    rotation_axis /= np.sqrt(np.sum(rotation_axis**2))
    rotation_angle = c_b.separation(c_t).to(u.rad).value
    rotmat = Rotation.from_rotvec(
        rotation_angle * rotation_axis).as_matrix()
    table_t['RA'], table_t['DEC'] = hp.rotator.rotateDirection(
        rotmat, table_t['RA'], table_t['DEC'], lonlat=True)
    table_t['RA'] = np.where(table_t['RA'] < 0, table_t['RA'] + 360,
                             table_t['RA'])

    # Rotate tiles additionally by hand to maximize overlap.
    rotmat = Rotation.from_rotvec(vec_b * 1.6).as_matrix()
    table_t['RA'], table_t['DEC'] = hp.rotator.rotateDirection(
        rotmat, table_t['RA'], table_t['DEC'], lonlat=True)
    table_t['RA'] = np.where(table_t['RA'] < 0, table_t['RA'] + 360,
                             table_t['RA'])

    hmap = hp.ud_grade(in_inner_buzzard, 512,
                       order_in='NESTED', order_out='NESTED')
    ra, dec = hp.pix2ang(512, np.arange(len(hmap))[hmap], nest=True,
                         lonlat=True)
    c_p = SkyCoord(ra, dec, unit='deg')

    c_t = SkyCoord(table_t['RA'], table_t['DEC'], unit='deg')
    sep2d = c_t.match_to_catalog_sky(c_p)[1]
    table_t = table_t[sep2d < 2 * u.deg]

    fig = plt.figure(figsize=(7.0, 3.5))
    crit = skm.stdDistortion
    proj = skm.Albers.optimize(ra_p[in_buzzard], dec_p[in_buzzard], crit=crit)
    footprint = skm.Map(proj, facecolor='white', ax=fig.gca())
    footprint.vertex(vertices[in_buzzard], facecolor='grey')
    footprint.vertex(vertices[in_inner_buzzard], facecolor='darkgrey')
    footprint.scatter(table_t['RA'], table_t['DEC'], s=0.1)
    footprint.focus(ra_p[in_buzzard], dec_p[in_buzzard])
    footprint.savefig('footprint.pdf')
    footprint.savefig('footprint.png', dpi=300)

    table_t[table_t['PROGRAM'] == 'BRIGHT'].write(
        os.path.join('buzzard-4', 'bright_tiles.fits'), overwrite=True)
    table_t[table_t['PROGRAM'] == 'DARK'].write(
        os.path.join('buzzard-4', 'dark_tiles.fits'), overwrite=True)


if __name__ == "__main__":
    main()
