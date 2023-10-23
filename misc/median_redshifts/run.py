import numpy as np
import zebu

z_med = dict()

for survey in ['bgs', 'lrg', 'des', 'hsc', 'kids']:

    table = zebu.read_mock_catalog(
        survey,  zebu.MOCK_PATH / 'buzzard-4', zebu.PIXELS)
    z_med[survey] = []

    if survey in ['bgs', 'lrg']:
        bins = zebu.LENS_Z_BINS[survey]
        table['z_true'] = table['z']
    else:
        bins = zebu.SOURCE_Z_BINS[survey]

    for tomographic_bin in range(len(bins) - 1):
        use = ((bins[tomographic_bin] < table['z']) &
               (table['z'] <= bins[tomographic_bin + 1]))
        z_med[survey].append(np.median(table['z_true'][use]))

print(z_med)
