#! /bin/bash

(
echo '#define GADGET_COMPILER_SETTINGS "" \'
echo "\" ${MPICC} \n \"\\"
echo "\" ${OPTIMIZE} \n \"\\"
for i in ${OPT}; do 
    echo "\" $i \n \"\\"
done
echo '""'

GIT=`git describe --always --dirty --abbrev=10`
echo '#define GADGET_VERSION "5.0.'${GIT}'"'
echo '#define GADGET_TESTDATA_ROOT "'$PWD'"'
) > $1


