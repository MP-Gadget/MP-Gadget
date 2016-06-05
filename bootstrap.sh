#! /bin/bash

git submodule init
git submodule update

cat <<EOF
If there were no errors fetching dependency submodules
then it is time to build MP-Gadget.

copy Options.mk.example to Opitions.mk,

and modify the file 

then type

make

EOF
