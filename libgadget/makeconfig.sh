#! /bin/bash

(
echo '#define GADGET_COMPILER_SETTINGS "" \'
echo "\" ${MPICC} \n \"\\"
echo "\" ${OPTIMIZE} \n \"\\"
for i in ${OPT}; do 
    echo "\" $i \n \"\\"
done
echo '""'

if [[ $VERSION = *dev* ]]; then
GIT=`git describe --always --dirty --abbrev=10`
VERSION=${VERSION}_${GIT/-/_}
fi
echo '#define GADGET_VERSION "'${VERSION}'"'
echo '#define GADGET_TESTDATA_ROOT "'$PWD'/.."'
) > $1


