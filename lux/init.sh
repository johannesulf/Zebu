module load openmpi
module load python/3.6.7
module load numpy
module load h5py
module load blas

export PYTHONPATH="${PYTHONPATH}:/data/groups/leauthaud/jolange/Zebu/"

source env/bin/activate

export HDF5_USE_FILE_LOCKING='FALSE'
