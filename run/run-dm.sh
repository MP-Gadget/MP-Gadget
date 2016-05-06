export OMP_NUM_THREADS=2
mpirun -np 4 ../GENIC/MP-GenIC paramfile-dm.genic
mpirun -np 8 ../MP-Gadget paramfile.gadget
