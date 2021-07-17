import zebu
import numpy as np

# %%

table_s = zebu.read_raw_data(1, 'source', 2, survey='hsc')

# %%

print('(1 + <m>)/(1 - <e_rms^2>): {:.5f}'.format(
    (1 + np.mean(table_s['m'])) / (1 - np.mean(table_s['sigma_rms']**2))))
print('<(1 + m)/(1 - e_rms^2)>: {:.5f}'.format(
    np.mean((1 + table_s['m']) / (1 - table_s['sigma_rms']**2))))
