= MP-Gadget3 =

Massvively parallel version of P-Gadget3.

== Pre-Installation ==

== Installation ==

```
git clone https://github.com/rainwoodman/MP-Gadget3

cd MP-Gadget3
git submodule init
git submodule update

```

We will need hdf5 and gsl. They are quite standard libraries.
usually can be loaded with 

```
module load hdf5 gsl
env |grep HDF
env |grep GSL
```
copy Options.mk.example to Options.mk

edit the file and set HDF / GSL flags according to the environment variables.

On coma, use Options.mk.example.coma, and run the following commands to load HDF + GSL
```
source ~yfeng1/local/bin/setup.sh
```

Edit Options.mk and tweak the compilation flags. We are in the process of cleaning this up.
Most options are tricky and undocumented. Talk to Yu Feng.

The defaults shall work for most cases; it enables Pressure-Entropy SPH and Blackhole, Cooling
and SFR. To run a N-Body sim, use IC files with no Gas particles.

```
make -j 8
```

It takes some time to build pfft, one of the bundled dependencies. 

Other libraries are bigfile and mp-sort, which are written by me and quick to build. 

For off-tree build, set DESTDIR in Options.mk; the default target is in build/

In the end, we will have 2 binaries:

build/P-Gadget3 and build/GENIC/N-GenIC

P-Gadget3 is the main simulation program.

N-GenIC is the initial condition generator.

== Usage: parameter files ==

There are two example runs in run/. 

    run.sh : simulation with gas
    run-dm.sh : simulation without gas (dm only)

== IO Format ==

The snapshot is in bigfile format. For data analysis in Python, use

```
   pip install bigfile
```

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)

