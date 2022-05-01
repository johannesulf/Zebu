#!/bin/bash
#SBATCH --qos=shared
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=haswell
#SBATCH --account=desi
#SBATCH --mem-per-cpu=16G

source /project/projectdirs/desi/software/desi_environment.sh master
module swap fiberassign/2.3.0
cd Zebu/mocks
cp ../misc/mock_footprint/bright_tiles_mock.fits bright/tiles.fits
cp ../misc/mock_footprint/dark_tiles_mock.fits dark/tiles.fits
fba_run --targets bright/targets.fits --footprint bright/tiles.fits --dir bright --rundate '2021-04-06T00:39:37'
fba_run --targets dark/targets.fits --footprint dark/tiles.fits --dir dark --rundate '2021-04-06T00:39:37'
cd ../..
PYTHONPATH=$PYTHONPATH:/project/projectdirs/desi/users/julange/LSS/py
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/bright/targets.fits --tiles Zebu/mocks/bright/tiles.fits --format fits --outdir Zebu/mocks/bright --realizations 64
srun -N 8 -c 8 -C haswell -A desi --qos=interactive -t 0:30:0 LSS/bin/mpi_bitweights --mtl Zebu/mocks/dark/targets.fits --tiles Zebu/mocks/dark/tiles.fits --format fits --outdir Zebu/mocks/dark --realizations 64
