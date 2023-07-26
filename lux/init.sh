module load openmpi
module load python/3.8.6
module load blas
module load intel/impi
module load intel/ifort

export PYTHONPATH="${PYTHONPATH}:/data/groups/leauthaud/jolange/Zebu/"

source env/bin/activate

export HDF5_USE_FILE_LOCKING='FALSE'

