import healpy as hp
import matplotlib.pyplot as plt
import numpy as np
import skymapper as skm

from pathlib import Path


def main():

    path = Path('buzzard-4')

    pixel_all = []
    for filepath in path.iterdir():
        if filepath.stem[:6] == 'pixel_':
            pixel_all.append(int(filepath.stem[6:]))

    nside = 8
    in_buzzard = np.isin(range(hp.nside2npix(nside)), pixel_all)
    ra_p, dec_p, vertices = skm.healpix.getGrid(nside, return_vertices=True,
                                                nest=True)[1:4]

    pixel_inner = []
    for pixel in pixel_all:
        neighbors = hp.get_all_neighbours(nside, pixel, nest=True)
        neighbors = neighbors[neighbors != -1]
        neighbors = hp.get_all_neighbours(
            nside, neighbors, nest=True).flatten()
        neighbors = neighbors[neighbors != -1]
        if np.all(np.isin(neighbors, pixel_all)):
            pixel_inner.append(pixel)

    in_inner_buzzard = np.isin(range(hp.nside2npix(nside)), pixel_inner)
    np.savetxt('pixels.csv', pixel_inner, fmt='%d')

    hmap = hp.ud_grade(in_inner_buzzard, 512,
                       order_in='NESTED', order_out='NESTED')
    ra, dec = hp.pix2ang(512, np.arange(len(hmap))[hmap], nest=True,
                         lonlat=True)

    fig = plt.figure(figsize=(7.0, 3.5))
    crit = skm.stdDistortion
    proj = skm.Albers.optimize(ra_p[in_buzzard], dec_p[in_buzzard], crit=crit)
    footprint = skm.Map(proj, facecolor='white', ax=fig.gca())
    footprint.vertex(vertices[in_buzzard], facecolor='grey')
    footprint.vertex(vertices[in_inner_buzzard], facecolor='darkgrey')
    footprint.focus(ra_p[in_buzzard], dec_p[in_buzzard])
    footprint.savefig('footprint.pdf')
    footprint.savefig('footprint.png', dpi=300)


if __name__ == "__main__":
    main()
