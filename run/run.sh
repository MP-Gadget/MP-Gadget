export OMP_NUM_THREADS=1
mpirun -np 4 ../build/MP-GenIC paramfile.genic || exit 1
mpirun -np 4 ../build/MP-Gadget paramfile.gadget || exit 1
