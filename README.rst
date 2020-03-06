MP-Gadget
=========

Massively Parallel Cosmological SPH Simulation Software - MP-Gadget.

`Source code browser <https://mp-gadget.github.io/MP-Gadget/classes.html>`_
(may be slightly out-sync from current master branch)


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
- Massive neutrinos
- Dark energy
- ICs have species dependent density and velocity transfer functions
- Generic halo tracer particle seeding
- Various wind feedback and blackhole feedback models
- Various star formation criteria
- Primordial and metal cooling using updated recombination rates from the Sherwood simulation.
- Fluctuating UV background

Installation
------------

First time users:

.. code:: bash

    git clone https://github.com/MP-Gadget/MP-Gadget.git
    cd MP-Gadget

    bash bootstrap.sh

We will need gsl. On HPC systems with the modules command, 
usually it can be loaded with 

.. code:: bash

    module load gsl

    env | grep GSL  # check if GSL path is reasonable

On a common PC/Linux system, refer to your package vendor how to
install gsl and gsl-devel.

You need an Options.mk file. A good first default is Options.mk.example .
Copy Options.mk.example to Options.mk

.. code:: bash

    cp Options.mk.example Options.mk

For other systems you should use the customised Options.mk file in the
platform-options directory. For example, for Stampede 2 you should do:

.. code:: bash

    cp platform-options/Options.mk.stampede2 Options.mk

Compile-time options may be set in Options.mk. The remaining compile time options are generally only useful for development or debugging. All science options are set using a parameter file at runtime.

- DEBUG which enables various internal code consistency checks for debugging.
- VALGRIND which if set disables the internal memory allocator and allocates memory from the system. This is required for debugging memory allocation errors with valgrind of the address sanitizer.
- NO_ISEND_IRECV_IN_DOMAIN disables the use of asynchronous send and receive in our custom MPI_Alltoallv implementation, for buggy MPI libraries.
- NO_OPENMP_SPINLOCK disables the use of OpenMP spinlocks and thus effectively disables threading. Necessary for platforms which do not provide an OpenMP header, such as Mac.

If compilation fails with errors related to the GSL, you may also need to set the GSL_INC or GSL_LIB variables in Options.mk to the filesystem path containing the GSL headers and libraries.

To run a N-Body sim, use IC files with no gas particles.

Now we are ready to build

.. code:: bash

    make -j

It takes some time to build pfft, a bundled dependency for pencil-based fast Fourier transforms.
Other libraries are bigfile and mp-sort, which are written by Yu Feng and are quick to build. 

In the end, we will have 2 binaries:

.. code::

    ls gadget/MP-Gadget genic/MP-GenIC

1. MP-Gadget is the main simulation program.

2. MP-GenIC is the initial condition generator.

Config Files
------------

Most options are configured at runtime with options in the config files.
The meaning of these options are documented in the params.c files in
the gadget/ and genic/ subdirectories.

Usage
-----

Find examples in examples/.

- dm-only : Dark Matter only
- lya : Lyman Alpha only
- hydro : hydro
- small : hydro with low resolution

Control number of threads with `OMP_NUM_THREADS`.

User Guide
----------

A longer user guide in LaTeX can be found here:
https://www.overleaf.com/read/kzksrgnzhtnh

IO Format
---------

The snapshot is in bigfile format. For data analysis in Python, use

.. code:: bash

   pip install bigfile

Refer to https://github.com/rainwoodman/bigfile for usage.

Otherwise directly open the blocks with Fortran or C, noting the data-type
information and attributes in header and attrs files (in plain text)

GLIBC 2.22
----------

Cray updated their GLIBC to 2.22+ recently. 
A good move but it happens to be a buggy version of GLIBC:
https://sourceware.org/bugzilla/show_bug.cgi?id=19590
causing non-existing symbols like `_ZGVcN4v___log_finite`.
Adding `-lmvec -lmvec_nonshared` to GSL_LIBS works around the issue.

Contributors
------------

Gadget-2 was authored by Volker Springel.
The original P-GADGET3 was maintained by Volker Springel

MP-Gadget is maintained by Yu Feng and Simeon Bird.

Contributors to MP-Gadget include:

Nicholas Battaglia, Nishikanta Khandai, Karime Maamari, Chris Pederson and Lauren Anderson.

Citation
--------

We never get around to write a proper code paper on MP-Gadget.

For usage of the code, here is a DOI for this repository that you can cite

.. image:: https://zenodo.org/badge/24486904.svg
   :target: https://zenodo.org/badge/latestdoi/24486904

It helps us to keep track of uses.

Licence Issue
-------------

Most files are licensed under GPLv2+.

Please refer to the source files for details.


Status
------

master branch status:

.. image:: https://travis-ci.org/MP-Gadget/MP-Gadget.svg?branch=master
       :target: https://travis-ci.org/MP-Gadget/MP-Gadget
