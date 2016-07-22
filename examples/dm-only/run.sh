export OMP_NUM_THREADS=2
$ROOT=../../../build
mpirun -np 4 $ROOT/GENIC/MP-GenIC paramfile-dm.genic || exit 1
mpirun -np 8 $ROOT/MP-Gadget paramfile-dm.gadget || exit 1
