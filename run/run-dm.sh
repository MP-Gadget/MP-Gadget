export OMP_NUM_THREADS=2
mpirun -np 4 ../build/GENIC/MP-GenIC paramfile-dm.genic || exit 1
mpirun -np 8 ../build/MP-Gadget paramfile-dm.gadget || exit 1
