from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u

table = Table.read('/global/cfs/cdirs/desi/survey/ops/surveyops/' +
                   'trunk/ops/tiles-main.ecsv')
c = SkyCoord(table['RA'], table['DEC'], unit='deg')
table = table[c.separation(SkyCoord(180, 40, unit='deg')) < 80 * u.deg]
table['RUNDATE'] = '2021-04-06T00:39:37'
table['FIELDROT'] = 1.0
table['FA_HA'] = 0.0
table['OBSCONDITIONS'] = 15
table['IN_DESI'] = 1
table[table['PROGRAM'] == 'DARK'].write('dark_tiles_ngc.fits')
table[table['PROGRAM'] == 'BRIGHT'].write('bright_tiles_ngc.fits')
