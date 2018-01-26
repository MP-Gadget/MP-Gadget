#!/bin/sh -e

BASEDIR="$1"

cd $BASEDIR;
make install CLASS_VERSION=$2 DEST=$3
