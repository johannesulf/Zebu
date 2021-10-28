#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell

source /project/projectdirs/desi/software/desi_environment.sh master
module swap fiberassign/2.3.0
cd Zebu/mocks
mkdir region_1/bright
mkdir region_1/dark
mv region_1/targets_bright.fits region_1/bright/targets.fits
mv region_1/targets_dark.fits region_1/dark/targets.fits
mv region_1/tiles_bright.fits region_1/bright/tiles.fits
mv region_1/tiles_dark.fits region_1/dark/tiles.fits
fba_run --targets region_1/bright/targets.fits --footprint region_1/bright/tiles.fits --dir region_1/bright --rundate '2021-04-06T00:39:37'
fba_run --targets region_1/dark/targets.fits --footprint region_1/dark/tiles.fits --dir region_1/dark --rundate '2021-04-06T00:39:37'
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/region_1/bright/targets.fits --tiles Zebu/mocks/region_1/bright/tiles.fits --format fits --outdir Zebu/mocks/region_1/bright --realizations 64
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/region_1/dark/targets.fits --tiles Zebu/mocks/region_1/dark/tiles.fits --format fits --outdir Zebu/mocks/region_1/dark --realizations 64
mv region_1/bright/targeted.fits region_1/targeted_bright.fits
mv region_1/dark/targeted.fits region_1/targeted_dark.fits
