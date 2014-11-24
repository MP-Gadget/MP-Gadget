= MP-Gadget3 =

Massvively parallel version of P-Gadget3.

== Pre-Installation ==

== Installation ==
```
git clone https://github.com/rainwoodman/MP-Gadget3

cd MP-Gadget3
git submodules init

```

We will need hdf5 and gsl. They are quite standard libraries.
usually can be loaded with 

```
module load hdf5 gsl
env |grep HDF
env |grep GSL
```
copy Makefile.default to Makefile.Local

edit Makefile.Local and set HDF / GSL flags according to the environments.

On coma, use Makefile.Warp, and run the following commands to load HDF + GSL
```
source ~yfeng1/local/bin/setup.sh
```

```
copy Makefile.example Makefile
```

Edit Makefile and enable the flags. (Tricky and undocumented! talk to Yu Feng)

An important variable is SYSTYPE. We will include Makefile.$(SYSTYPE) for the
machine local settings that are unique to this machine.

On COMA and Warp use the same settings.

Otherwise, reference Makefile.default to build your own Makefile.Local file and
set SYSTYPE=Local in Makefile

The defaults shall work for most cases; it enables Pressure-Entropy SPH and Blackhole, Cooling
and SFR. To run a N-Body sim, use IC files with no Gas particles.

Finally build the submodules and MP-Gadget3.
```
make -j 8
```
It takes some time to build fftw3 and pfft. Other libraries are bigfile and
radixsort, which are written by me and really quick to build. 
In the end, we will have 2 binaries:

P-Gadget3 and GENIC/N-GenIC

P-Gadget3 is the main simulation program.
N-GenIC is the initial condition generator.

== Usage: parameter files ==

There are two example runs in run/. 

    run.sh : simulation with gas
    run-dm.sh : simulation without gas (dm only)

