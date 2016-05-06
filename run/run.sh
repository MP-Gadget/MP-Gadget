export OMP_NUM_THREADS=1
mpirun -np 4 ../build/GENIC/MP-GenIC paramfile.genic
mpirun -np 4 ../build/MP-Gadget paramfile.gadget
