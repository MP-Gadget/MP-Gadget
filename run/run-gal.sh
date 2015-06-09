export OMP_NUM_THREADS=1
mpirun -np 4 ../GENIC/N-GenIC paramfile.genic-late
mpirun -np 4 ../P-Gadget3 paramfile.gadget-gal
