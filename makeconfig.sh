#! /bin/bash

echo 'const char * COMPILETIMESETTINGS = "" '
echo "\" ${MPICC} \n \""
echo "\" ${OPTIMIZE} \n \""
for i in ${OPT}; do 
    echo "\" $i \n \""
done
echo ';'
GIT=`git describe --always --dirty --abbrev=10`
echo 'const char * GADGETVERSION = "5.0.'${GIT}'";'

