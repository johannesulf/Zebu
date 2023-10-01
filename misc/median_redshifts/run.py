import numpy as np
import zebu

z_s = dict()

for survey in ['des', 'hsc', 'kids']:

    table_s = zebu.read_mock_catalog(
        survey,  zebu.MOCK_PATH / 'buzzard-4', zebu.PIXELS)
    z_s[survey] = []

    for source_bin in range(len(zebu.SOURCE_Z_BINS[survey]) - 1):
        use = ((zebu.SOURCE_Z_BINS[survey][source_bin] < table_s['z']) &
               (table_s['z'] <= zebu.SOURCE_Z_BINS[survey][source_bin + 1]))
        z_s[survey].append(np.median(table_s['z_true'][use]))

print(z_s)
