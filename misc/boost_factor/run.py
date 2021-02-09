import os
import zebu
import numpy as np
import argparse
from astropy.table import Table
import matplotlib.pyplot as plt
from dsigma.stacking import shape_noise_error, boost_factor

# %%

parser = argparse.ArgumentParser()
parser.add_argument('survey', help='the lensing survey')
parser.add_argument('lens_bin', help='the lens bin', type=int)
args = parser.parse_args()

stage = 1

# %%

kwargs = {
        'photo_z_dilution_correction': True,
        'boost_correction': True, 'random_subtraction': True,
        'return_table': True,
        'shear_bias_correction': args.survey != 'des',
        'shear_responsivity_correction': args.survey == 'hsc',
        'metacalibration_response_correction': args.survey == 'des'}

rp = np.sqrt(zebu.rp_bins[1:] * zebu.rp_bins[:-1])

for source_bin in range(4):

    table_l = Table.read(os.path.join(
        zebu.base_dir, 'stage_1', 'precompute', 'l{}_s{}_{}_l.hdf5'.format(
            args.lens_bin, source_bin, args.survey)))
    table_r = Table.read(os.path.join(
        zebu.base_dir, 'stage_1', 'precompute', 'l{}_s{}_{}_r.hdf5'.format(
            args.lens_bin, source_bin, args.survey)))

    if np.sum(table_l['sum 1'] == 0):
        continue

    kwargs['table_r'] = table_r

    err = np.median(rp * shape_noise_error(table_l, **kwargs))
    b = boost_factor(table_l, table_r)

    source_z_bins = zebu.source_z_bins(stage, survey=args.survey)

    plt.plot(rp, b, label=r'${:.1f}\leq z_s \leq {:.1f}$, Error:{:.1f}'.format(
        source_z_bins[source_bin], source_z_bins[source_bin+1], err))

plt.title(r'{}, ${:.1f} \leq z_l \leq {:.1f}$'.format(
    args.survey.upper(), zebu.lens_z_bins[args.lens_bin],
    zebu.lens_z_bins[args.lens_bin+1]))
plt.xscale('log')
plt.ylim(0.9, 1.2)
plt.axhline(1.0, ls='--', color='black')
plt.legend(loc='best')
plt.xlabel(r'Projected Radius $r_p \, [h^{-1} \, \mathrm{Mpc}]$')
plt.ylabel(r'Traditional Boost Factor $b$')
plt.tight_layout(pad=0.3)
plt.savefig('{}_{}.pdf'.format(args.survey.upper(), args.lens_bin))
plt.savefig('{}_{}.png'.format(args.survey.upper(), args.lens_bin),
            dpi=300)
