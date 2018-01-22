MP-Gadget
=========

Massively Parallel Cosmological SPH Simulation Software - MP-Gadget.

`Source code browser <https://rainwoodman.github.io/MP-Gadget/classes.html>`_
(maybe slightely out-sync from current master branch)


Description
-----------

This version of Gadget is derived from main P-Gadget / Gadget-2. It is the source code
used to run the BlueTides simulation (http://bluetides-project.org).

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

    git clone https://github.com/bluetides-project/MP-Gadget
    # or git clone https://github.com/rainwoodman/MP-Gadget-private 
    cd MP-Gadget

    bash bootstrap.sh

We will need gsl. On HPC systems with the modules command, 
usually it can be loaded with 

.. code:: bash

    module load gsl

    env | grep GSL  # check if GSL path is reasonable

On a common PC/Linux system, refer to your package vendor how to
install gsl and gsl-devel.

Copy Options.mk.example to Options.mk

.. code:: bash

    cp Options.mk.example Options.mk

Edit Options.mk

1. Set GSL flags according to the environment variables.
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

    ls build/MP-Gadget build/MP-GenIC

1. MP-Gadget is the main simulation program.

2. MP-GenIC is the initial condition generator.

GLIBC 2.22
----------

Cray updated their GLIBC to 2.22+ recently. 
A good move but it happens to be a buggy version of GLIBC:
https://sourceware.org/bugzilla/show_bug.cgi?id=19590
causing non-existing symbols like `_ZGVcN4v___log_finite`.
Adding `-lmvec -lmvec_nonshared` to GSL_LIBS works around the issue.

Usage
-----

Find examples in examples/.

- dm-only : Dark Matter only
- lya : Lyman Alpha only (needs special compilcation flags)
- hydro : hydro
- small : hydro with low resolution

OpenMP Complication
-------------------

When OpenMP is switched from on to off or off to on,
all of the dependencies needs to be recompiled.
This can be achived by removing all files in depends/lib.

Otherwise symbols related to OpenMP in PFFT may be missing.

Always enable OpenMP.

IO Format
---------

The snapshot is in bigfile format. For data analysis in Python, use

.. code:: bash

   pip install bigfile

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)

Contributors
------------

Gadget-2 was authored by Volker Springel.
The original P-GADGET3 was maintained by Volker Springel

MP-Gadget is maintained by Yu Feng.

Contributors to MP-Gadget include:

Simeon Bird, Nicholas Battaglia, Nishikanta Khandai

Citation
--------

Please cite 'Feng et al 2016 in prep'. A short paper will be written soon.

We need to obtain a DOI for direct citation of the software.

Licence Issue
-------------

Most files are licensed under GPLv2+.

Except two files of questionable licences:
sfr_eff.c and cooling.c.

Please refer to the source files for details.

The source code is put in public domain for reference.
To enable features in sfr_eff.c and cooling.c for scientific runs,
consent from the original authors of these files shall be obtained.

Status
------

master branch on @rainwoodman:

.. image:: https://travis-ci.org/rainwoodman/MP-Gadget.svg?branch=public
       :target: https://travis-ci.org/rainwoodman/MP-Gadget
