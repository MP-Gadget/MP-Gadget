MP-Gadget
=========

Massively Parallel Cosmological SPH Simulation Software - MP-Gadget.

Description
-----------

This version of Gadget is derived from main P-Gadget / Gadget-2.

The infrastructure is heavily reworked. As a summary:

- A better PM solver for long range force with Pencil FFT.
- A better Tree solver with faster threading and less redundant code.
- A better Domain decomposition that scales to half a million cores.
- A easier to use IO module with a Python binding.
- A more intuitive parameter file parser with schema and docstrings.
- A cleaner code base with less conditional compilation flags.

Physics models:

- Pressure Entropy SPH and Density Entropy SPH
- Radiation background in the expansion
- Generic tracer particle seeding
- Various wind feedback and blackhole feedback models
- Various star formation criteria (need a License from Phil Hopkins before enabling)
- Primordial and metal cooling
- Fluctuating UV background

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

    env | grep HDF
    env | grep GSL

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

    ls build/MP-Gadget build/GENIC/MP-GenIC

1. MP-Gadget is the main simulation program.

2. MP-GenIC is the initial condition generator.

Usage
-----
There are two example runs in run/. 

    run.sh : simulation with gas
    run-dm.sh : simulation without gas (dm only)

OpenMP Complication
-------------------

When OpenMP is switched from on to off or off to on,
all of the dependencies needs to be recompiled.
This can be achived by removing all files in depends/lib.

Otherwise symbols related to OpenMP in PFFT may be missing.

IO Format
---------

The snapshot is in bigfile format. For data analysis in Python, use

.. code:: bash

   pip install bigfile

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)

Citation
--------

A code paper will be nice.
We need to obtain a DOI for direct citation of the software.


