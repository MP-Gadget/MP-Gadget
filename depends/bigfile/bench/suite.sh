IOSIM=../utils/bigfile-iosim

NP=$1
NFILE=$2
SIZE=$3

STR=$NP-$NFILE-$SIZE

mpirun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 create /tmp/bench-$STR
mpirun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 read /tmp/bench-$STR
mpirun -n $NP $IOSIM -f $NFILE -n $NP -s 1024 -w 1 -p update /tmp/bench-$STR
