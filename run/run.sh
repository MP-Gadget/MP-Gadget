export OMP_NUM_THREADS=2
mpirun -np 4 ../GENIC/N-GenIC paramfile.genic
mpirun -np 8 ../P-Gadget3 paramfile.gadget
