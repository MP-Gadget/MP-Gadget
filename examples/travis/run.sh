export OMP_NUM_THREADS=2
ROOT=../../build
mpirun -np 2 $ROOT/MP-GenIC paramfile.genic || exit 1
mpirun -np 2 $ROOT/MP-Gadget paramfile.gadget || exit 1
