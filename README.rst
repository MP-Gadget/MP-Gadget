bigfile
=======

A reproducible massively parallel IO library for large, hierarchical datasets.

Python 2 and 3 binding is available via pip.

:code:`bigfile` was originally developed for the BlueTides simulation 
on BlueWaters at NCSA. The library is currently under investigation under the
BW-PAID program with NCSA.

The current implementation works on a true POSIX compliant file system, e.g. Lustre.
BigFile makes two assumptions

1) mkdir() is durable -- it shall propagate the directory to all clients.

2) Allowing non-overlapping write from different clients. This less strict than POSIX.

Be aware NFS is not a true POSIX compliant file system.

To cite bigfile, use DOI at

.. image:: https://zenodo.org/badge/21016779.svg
   :target: https://zenodo.org/badge/latestdoi/21016779

Build status
------------
.. image:: https://github.com/rainwoodman/bigfile/workflows/main/badge.svg
    :alt: Build Status
    :target: https://github.com/rainwoodman/bigfile/actions?query=workflow%3Amain

.. image:: https://scan.coverity.com/projects/11368/badge.svg
    :alt: Coverity Scan Build Status
    :target: https://scan.coverity.com/projects/rainwoodman-bigfile

Install
-------

Usually one only needs the Python binding in order to read a BigFile.

To install the Python binding

.. code:: bash

    pip install [--user] bigfile

The C-API of bigfile can be embedded into a project, by dropping in 
four files : bigfile.c bigfile-mpi.c, bigfile.h bigfile-mpi.h.

However, if installation is preferred, the library and executables can be compiled and installed
using CMake:

.. code:: bash
    
    mkdir build
    cd build
    cmake ..
    make install
    
This will install the project in the default path (probalby /usr/local), to select an alternative
installation destination, replace the cmake call by:

.. code:: bash

    cmake -DCMAKE_INSTALL_PREFIX:PATH=<PREFIX> ..
    
where <PREFIX> is the desired destination.

Compilation is also possible using the legacy build system:

.. code:: bash

    make install

However, you need to manually override CC MPICC, PREFIX as needed. Take a look at the Makefile is always recommended.


Description
-----------

:code:`bigfile` provides a hierarchical structure of data columns via 
:code:`File`, :code:`Dataset` and :code:`Column`. 

A :code:`Column` stores a two dimesional table of :code:`nmemb` columns 
and :code:`size` rows. Numerical typed columns are supported.

Attributes can be attached to a :code:`Column`. 
Numerical attributes and string attributes are supported.

Type casting is performed on-the-fly if read/write operation requests a different data type than the file has stored.

:code:`bigfile.Dataset` works with `dask.from_array <http://dask.pydata.org>`_.

The Anatomy of a BigFile
++++++++++++++++++++++++

A `BigFile` maps to a directory hierarchy on the file system.

This is the directroy structure of an example file:

.. code::

    /scratch1/scratchdirs/sd/yfeng1/example-bigfile
      block0
        header
        attrs-v2
        000000
      group1
        block1.1
          header
          attrs-v2
          000000
          000001
        block1.2
          header
          attrs-v2
          000000
          000001
      group2
        block2.1 
          header
          attrs-v2
          000000
          000001

A `BigFile` consists of blocks (`BigBlock`) and groups of blocks. 
Files, groups and blocks are mapped to directories of the hosting file system.

A `BigBlock` consists of two special plain text files and a sequence of binary data files.

- Text file :code:`header`, which stores the data type and size of the block,
- Text file :code:`attrs-v2`, which stores the attributes attached to the block.
- Binary files :code:`000000`, :code:`000001`, .... which store the binary data
  of for the blocks. The format of the data (endianess, data type, vector length per row)
  is described in `header`. The number of files used by a block, as well as the size
  (number of rows) of a block is fixed at the creation of a block. 

The performance of bigfile is insulated from the configurations of 
the Lustre file system due to the explicit striping.

Comparision with HDF5
---------------------

**Good**

- bigfile is simpler. The core library of bigfile consists of 2 source files, 2 header
  files, and 1 Makefile, a total of less than 3000 lines of code, 
  easily maintained by one person or dropped into a project. 
  HDF5 is much more complicated.

- bigfile is closer to the data. The raw data on disk is stored as binary files
  that can be directly accessed by any application. The meta data (block 
  descriptions and attributes) is stored in plain text, easily understood by
  human. In a sense, the :code:`bigfile` library is no more than a helper 
  for reading and writing these files under the bigfile protocal. 
  In contrast, once your data goes into  HDF5 it is trapped, 
  the HDF5 library is required to make sense of the data from that point on.

**Bad**

- bigfile is limited -- for example, bigfile has no API for output streaming,
  and only 2-dimensional tables are supported.
  HDF5 is much richer in functionality and more powerful in data description.  
  The designated use-case of bigfile is to store 
  a large amount of static / near-immutable column-wise table data. 

- bigfile is incomplete. Bugs have yet to be identified and fixed.  
  In contrast HDF5 has been a funded research program developed for more than 20 years. 

API Reference
-------------

The documentation needs to be written.

The core library is C.  Refer to bigfile.h and bigfile-mpi.h for the API interface.

There are Python bindings for Python 2 and 3.

The Python binding under MPI invoked more meta-data queries to the file system
than we would like to be, though for small scale applications (thousands of cores)
it is usually adequate.

Examples
++++++++

.. code:: python

    # This example consumes the BlueTides Simulation data.

    import bigfile

    f = bigfile.File('PART_018')

    print (f.blocks)
    # Position and Velocity of GAS particles
    data = bigfile.Dataset(f["0/"], ['Position', 'Velocity'])
    
    print (data.size)
    print (data.dtype)
    # just read a few particles, because there are 700 billion of them.
    print data[10:30]

    
Shell
-----

We provide the following shell commands for inspecting a bigfile:

- bigfile-cat
- bigfile-create
- bigfile-repartition
- bigfile-ls
- bigfile-get-attr
- bigfile-set-attr

Rejected Poster for SC17
------------------------

We submitted a poster to describe bigfile for SC17. Although the poster was rejected, we post them
here as they contain a description of the design and some benchmarks of bigfile.

    https://github.com/rainwoodman/bigfile/tree/documents

Yu Feng
