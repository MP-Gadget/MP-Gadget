export OMP_NUM_THREADS=1
mpirun -np 4 ../build/GENIC/N-GenIC paramfile.genic
mpirun -np 4 ../build/P-Gadget3 paramfile.gadget
