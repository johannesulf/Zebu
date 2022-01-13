#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell
#SBATCH --account=desi

source /project/projectdirs/desi/software/desi_environment.sh master
module swap fiberassign/2.3.0
cd Zebu/mocks
mkdir mocks/bright
mkdir mocks/dark
mv mocks/targets_bgs.fits mocks/bright/targets.fits
mv mocks/targets_lrg.fits mocks/dark/targets.fits
mv ../misc/mock_footprint/bright_tiles_mock.fits mocks/bright/tiles.fits
mv ../misc/mock_footprint/dark_tiles_mock.fits mocks/dark/tiles.fits
fba_run --targets mocks/bright/targets.fits --footprint mocks/bright/tiles.fits --dir mocks/bright --rundate '2021-04-06T00:39:37'
fba_run --targets mocks/dark/targets.fits --footprint mocks/dark/tiles.fits --dir mocks/dark --rundate '2021-04-06T00:39:37'
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/mocks/bright/targets.fits --tiles Zebu/mocks/mocks/bright/tiles.fits --format fits --outdir Zebu/mocks/mocks/bright --realizations 64
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/mocks/dark/targets.fits --tiles Zebu/mocks/mocks/dark/tiles.fits --format fits --outdir Zebu/mocks/mocks/dark --realizations 64
mv mocks/bright/targeted.fits mocks/targeted_bright.fits
mv mocks/dark/targeted.fits mocks/targeted_dark.fits
