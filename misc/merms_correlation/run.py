import zebu
import numpy as np

# %%

table_s = zebu.read_mock_catalog('hsc', zebu.MOCK_PATH / 'buzzard-4',
                                 zebu.PIXELS)

# %%

print('1 / (1 + <m>) (1 - <e_rms^2>): {:.5f}'.format(
    1.0 / (1 + np.mean(table_s['m'])) / (1 - np.mean(table_s['e_rms']**2))))
print('1 / <(1 + m) (1 - e_rms^2)>: {:.5f}'.format(
    1.0 / np.mean((1 + table_s['m']) * (1 - table_s['e_rms']**2))))