#! /bin/bash

echo 'const char * COMPILETIMESETTINGS = "" '
echo \" $OPTIMIZE \n \"
for i in $OPT; do 
    echo \" $i \\\n \"
done
echo ';'

