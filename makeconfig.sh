#! /bin/bash

echo 'const char * COMPILETIMESETTINGS = "" '
echo "\" ${CC} \n \""
echo "\" ${OPTIMIZE} \n \""
for i in ${OPT}; do 
    echo "\" $i \n \""
done
echo ';'

