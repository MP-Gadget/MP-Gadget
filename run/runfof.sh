export OMP_NUM_THREADS=1
mpirun -np 4 ../build/P-Gadget3 paramfile.fof 3 003

cd fofdebug
python assert.py
