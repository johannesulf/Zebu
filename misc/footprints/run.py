import numpy as np
import skymapper as skm
import matplotlib.pyplot as plt
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy import units as u

fig = plt.figure(figsize=(7, 3.5))
proj = skm.Hammer()
footprint = skm.Map(skm.Hammer(), facecolor='white', ax=fig.gca())
sep = 30
footprint.grid(sep=sep)

nside = 1024

des = skm.survey.DES()
vertex = footprint.footprint(des, nside, facecolor='red', alpha=0.4)
footprint.ax.text(0, np.deg2rad(-70), 'DES', color='red',
                  horizontalalignment='center', verticalalignment='bottom')

pixels, rap, decp, vertices = skm.healpix.getGrid(nside, return_vertices=True)
# http://kids.strw.leidenuniv.nl/overview.php
ra_min_list = [0, 329.5, 156, 225, 128.5]
ra_max_list = [53.5, 360, 225, 238, 141.5]
dec_min_list = [-35.6, -35.6, -5, -3, -2]
dec_max_list = [-25.7, -25.7, +4, +4, +3]
inside = np.zeros(len(pixels), dtype=bool)
for ra_min, ra_max, dec_min, dec_max in zip(ra_min_list, ra_max_list,
                                            dec_min_list, dec_max_list):
    inside = inside | ((ra_min <= rap) & (rap <= ra_max) & (dec_min <= decp) &
                       (decp <= dec_max))
footprint.vertex(vertices[inside], facecolor='purple', alpha=0.6)
footprint.ax.text(np.deg2rad(29), np.deg2rad(-26), 'KiDS', color='purple',
                  horizontalalignment='right', verticalalignment='bottom')

# https://hsc.mtk.nao.ac.jp/ssp/wp-content/uploads/2016/05/hsc_ssp_rv_jan13.pdf
ra_min_list = [330, 0, 27.5, 127.5, 200]
ra_max_list = [360, 40, 40, 225, 250]
dec_min_list = [-1, -1, -7, -2, +42.5]
dec_max_list = [+7, +7, +1, +5, +44]
inside = np.zeros(len(pixels), dtype=bool)
for ra_min, ra_max, dec_min, dec_max in zip(ra_min_list, ra_max_list,
                                            dec_min_list, dec_max_list):
    inside = inside | ((ra_min <= rap) & (rap <= ra_max) & (dec_min <= decp) &
                       (decp <= dec_max))
footprint.vertex(vertices[inside], facecolor='royalblue', alpha=0.4)
footprint.ax.text(np.deg2rad(-15), np.deg2rad(+7), 'HSC', color='royalblue',
                  horizontalalignment='center', verticalalignment='bottom')

# https://www.legacysurvey.org/dr9/files/
desi = vstack([Table.read('survey-bricks-dr9-north.fits.gz'),
               Table.read('survey-bricks-dr9-south.fits.gz')])
desi = desi[desi['in_desi']]

c_desi = SkyCoord(desi['ra'], desi['dec'], unit='deg')
c_footprint = SkyCoord(rap, decp, unit='deg')
idx, sep2d, dist3d = c_footprint.match_to_catalog_sky(c_desi)
inside = sep2d < 0.4 * u.deg
footprint.vertex(vertices[inside], facecolor='grey', alpha=0.4)
footprint.ax.text(0, np.deg2rad(+32), 'DESI', color='grey',
                  horizontalalignment='center', verticalalignment='bottom')

footprint.savefig('footprint.pdf')
footprint.savefig('footprint.png', dpi=300, transparent=True)
