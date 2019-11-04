export OMP_NUM_THREADS=1
ROOT=../../
mpirun -np 12 $ROOT/genic/MP-GenIC paramfile.genic || exit 1
mpirun -np 12 $ROOT/gadget/MP-Gadget paramfile.gadget || exit 1
