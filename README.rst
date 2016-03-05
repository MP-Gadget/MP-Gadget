MP-Gadget3
==========

Massvively parallel version of P-Gadget3.

Description
-----------

Installation
------------

First time users:

.. code:: bash

    git clone https://github.com/rainwoodman/MP-Gadget3

    cd MP-Gadget3
    git submodule init
    git submodule update

We will need hdf5 and gsl. They are quite standard libraries.
usually can be loaded with 

.. code:: bash

    module load hdf5 gsl

    env |grep HDF
    env |grep GSL

Copy Options.mk.example to Options.mk

.. code:: bash

    cp Options.mk.example Options.mk

Edit Options.mk

1. Set HDF / GSL flags according to the environment variables.
   On coma, use Options.mk.example.coma.

2. Tweak the compilation options for 'features'. 
   We are in the process of cleaning this up.
   Most options are tricky and undocumented, as Gadget.
   The defaults shall work for most cases; 
   it enables Pressure-Entropy SPH and Blackhole, Cooling
   and SFR. To run a N-Body sim, use IC files with no gas particles.

3. For off-tree build, set DESTDIR in Options.mk; the default target is in build/

Now we are ready to build

.. code:: bash

    make -j 8

It takes some time to build pfft, one of the bundled dependencies. 
Other libraries are bigfile and mp-sort, which are written by me and quick to build. 

In the end, we will have 2 binaries:

.. code::

    ls build/P-Gadget3 build/GENIC/N-GenIC

1. P-Gadget3 is the main simulation program.

2. N-GenIC is the initial condition generator.

Usage
-----
There are two example runs in run/. 

    run.sh : simulation with gas
    run-dm.sh : simulation without gas (dm only)

IO Format
---------

The snapshot is in bigfile format. For data analysis in Python, use

.. code:: bash

   pip install bigfile

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)

