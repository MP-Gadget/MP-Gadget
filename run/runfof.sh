export OMP_NUM_THREADS=1
mpirun -np 4 ../build/MP-Gadget paramfile.fof 3 003

cd fofdebug
python assert.py
