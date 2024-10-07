#/bin/bash
trap exit ERR

# this script is used to create a new build of MP-Gadget
# for any given rev/branch. branch will be resolved to a rev first.
# the directory MP-Gadget must be a cloned repo with origin point
# to the remote.

# copy / link the script to the correct path.

if [ "x$1" == "x" ]; then
    echo usage : $0 rev host
    echo host is used to find platform-options/Options.mk
    echo example : $0 master coriknl
    exit 1
fi

rev=$1
host=$2

# side effect: sets codedir
function findrev {
    local rev=$1

    # no stdout allowed
    pushd MP-Gadget 1>&2

    git fetch origin 1>&2
    sha=`git describe --always --abbrev=10 $rev`

    popd 1>&2

    echo $sha
}

function checkout {
    local codedir=$1
    local rev=$2
    echo --------- checking $rev as $codedir ----- 
    if ! [ -d $codedir ]; then
        cp -R MP-Gadget $codedir
    fi

    pushd $codedir

    git checkout -f $rev

    popd
}


function build {
    local codedir=$1
    echo --------- building $codedir ----- 
    echo $codedir
    pushd $codedir

    if ! [ -f Options.mk ]; then
        cp platform-options/Options.mk.$host Options.mk
        ./bootstrap.sh
    fi
    make

    popd
}

codedir=$host-`findrev $rev`
echo $codedir
checkout $codedir $rev

build $codedir



