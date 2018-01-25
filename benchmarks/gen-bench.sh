#! /bin/bash
#
# this script generates a benchmark from current suite to
# a target prefix.
# it will copy all files and replace @PREFIX@ to the directory.

if [ "-x$2" == "x" ]; then
    echo Usage $0 benchdir  prefix
    exit 1;
fi

bench=$1
prefix=$2

mkdir -p $prefix

cd $bench

for i in *; do
    sed -e "s;@PREFIX@;$prefix;" $i > $prefix/$i ;
done

