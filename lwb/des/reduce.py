import h5py
import numpy as np
from astropy.table import Table

# %%

table_s = Table()

fstream = h5py.File('DESY3_sompz_v0.40.h5')
table_s['bhat'] = fstream['catalog/sompz/unsheared/bhat'][()]
fstream.close()

fstream = h5py.File('DESY3_metacal_v03-004.h5')

for key in ['ra', 'dec', 'e_1', 'e_2', 'R11', 'R12', 'R21', 'R22', 'weight']:
    table_s[key] = fstream['catalog/unsheared/' + key][()]

for sheared in ['1m', '1p', '2m', '2p']:
    table_s['weight_{}'.format(sheared)] = fstream[
        'catalog/sheared_{}/weight'.format(sheared)][()]

fstream.close()


fstream = h5py.File('DESY3_indexcat.h5')

for flag in ['select', 'select_1p', 'select_1m', 'select_2p', 'select_2m']:
    table_s['flags_' + flag] = np.zeros(len(table_s), dtype=bool)
    table_s['flags_' + flag][fstream['index/' + flag][()]] = True

select = (table_s['flags_select'] | table_s['flags_select_1p'] |
          table_s['flags_select_1m'] | table_s['flags_select_2p'] |
          table_s['flags_select_2m']) & (table_s['bhat'] >= 0)
table_s = table_s[select]
fstream.close()

# %%

select = (table_s['ra'] < 60) | (table_s['ra'] > 300)
select &= table_s['dec'] > -22.5
table_s = table_s[select]
table_s.write('des_y3.hdf5', path='catalog', overwrite=True)

# %%

table_n = Table.read(
    '2pt_NG_final_2ptunblind_02_26_21_wnz_maglim_covupdate.fits', hdu=6)
table_n.rename_column('Z_MID', 'z')
table_n['n'] = np.vstack([table_n['BIN{}'.format(i + 1)] for i in range(4)]).T
table_n.keep_columns(['z', 'n'])
table_n.write('des_y3.hdf5', path='redshift', overwrite=True, append=True)
