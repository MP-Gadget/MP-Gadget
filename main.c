#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include <fftw3-mpi.h>
#ifdef INVARIANCETEST
#define DO_NOT_REDEFINE_MPI_COMM_WORLD
#endif

#include "allvars.h"
#include "proto.h"



/*! \file main.c
 *  \brief start of the program
 */
/*!
 *  This function initializes the MPI communication packages, and sets
 *  cpu-time counters to 0. Then begrun() is called, which sets up
 *  the simulation either from IC's or from restart files.  Finally,
 *  run() is started, the main simulation loop, which iterates over
 *  the timesteps.
 */
int main(int argc, char **argv)
{
  int i;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  fftw_init_threads();
  fftw_mpi_init();
  fftw_plan_with_nthreads(omp_get_max_threads());

#ifdef INVARIANCETEST
  World_ThisTask = ThisTask;
  World_NTask = NTask;

  if(World_NTask != (INVARIANCETEST_SIZE1 + INVARIANCETEST_SIZE2))
    {
      printf("wrong number of procs for invariance-test\n");
      MPI_Finalize();		/* clean up & finalize MPI */
      return 0;
    }

  if(World_ThisTask < INVARIANCETEST_SIZE1)
    Color = 0;
  else
    Color = 1;

  MPI_Comm_split(MPI_COMM_WORLD, Color, 0, &MPI_CommLocal);
  MPI_Comm_rank(MPI_CommLocal, &ThisTask);
  MPI_Comm_size(MPI_CommLocal, &NTask);
#endif

  for(PTask = 0; NTask > (1 << PTask); PTask++);

  if(argc < 2)
    {
      if(ThisTask == 0)
	{
	  printf("Parameters are missing.\n");
	  printf("Call with <ParameterFile> [<RestartFlag>] [<RestartSnapNum>]\n");
	  printf("\n");
	  printf("   RestartFlag    Action\n");
	  printf("       0          Read iniial conditions and start simulation\n");
	  printf("       1          Read restart files and resume simulation\n");
	  printf("       2          Restart from specified snapshot dump and continue simulation\n");
	  printf("       3          Run FOF and optionally SUBFIND if enabled\n");
	  printf("       4          Convert snapshot file to different format\n");
	  printf("\n");
	}
      endrun(0);
    }

  strcpy(ParameterFile, argv[1]);

  if(argc >= 3)
    RestartFlag = atoi(argv[2]);
  else
    RestartFlag = 0;

  if(argc >= 4)
    RestartSnapNum = atoi(argv[3]);
  else
    RestartSnapNum = -1;

  begrun();			/* set-up run  */

  run();			/* main simulation loop */

  MPI_Finalize();		/* clean up & finalize MPI */

  return 0;
}




/* ----------------------------------------------------------------------
   The rest of this file contains documentation for compiling and
   running the code, in a format appropriate for 'doxygen'.
   ----------------------------------------------------------------------
 */

/*! \mainpage Reference documentation of GADGET-2

\author Volker Springel \n
        Max-Planck-Institute for Astrophysics \n
        Garching, Germany \n
        volker@mpa-garching.mpg.de \n

\n

\section install Compilation 

GADGET-2 needs the following non-standard libraries for compilation:

- \b MPI - the Message Passing Interface (version 1.0 or higher). Many
  vendor supplied versions exist, in addition to excellent open source
  implementations, e.g.  MPICH
  (http://www-unix.mcs.anl.gov/mpi/mpich/) or LAM
  (http://www.lam-mpi.org/).
- \b GSL - the GNU scientific library. This open-source package can be
  obtained at http://www.gnu.org/software/gsl, for example. GADGET
  only needs this library for a few very simple cosmological
  integrations at start-up.
- \b FFTW - the <em>Fastest Fourier Transform in the West</em>. This
  open-source package can be obtained at http://www.fftw.org,
  for example. It is only needed for simulations that use the TreePM algorithm.

Note that FFTW needs to be compiled with parallel support enabled.
This can be achieved by passing the option <b>--enable-mpi</b> to the
configure script. When at it, you might as well add
<b>--enable-type-prefix</b> to obtain the libraries in both a single
and double precision version. If this has not been done, you should
set the option \b NOTYPEPREFIX_FFTW in GADGET's \ref Gadget-Makefile
"Makefile".

Note that if any of the above libraries is not installed in standard
locations on your system, the \ref Gadget-Makefile "Makefile" provided
with the code may need slight adjustments. Similarly, compiler
options, particularly with respect to optimisations, may need
adjustment to the C-compiler that is used. Finally, the \ref
Gadget-Makefile "Makefile" contains a number of compile-time options
that need to be set appropriately for the type of simulation that is
simulated.

The provided makefile is compatible with GNU-make, i.e. typing \b make
or \b gmake should then build the executable <b>P-Gadget2<b>.  If your
site does not have GNU-make, get it, or write your own makefile.

\section howtorun Running the code

In order to start the code, a \ref parameterfile "parameterfile" needs
to be specified.

*/











/*! \page parameterfile  Parameterfile of GADGET-2

- \b ViscositySourceScaling \n Parameter for dynamic viscosity (e.g. 2.5)

- \b ViscosityDecayLength \n Parameter for dynamic viscosity (e.g. 2.0)

- \b ViscosityAlphaMin \n  Parameter for dynamic viscosity (e.g. 0.01)

- \b ViscosityAlphaMin \n  value for w of equation of state (e.g. -0.4)

- \b DarkEnergyFile \n  file containing z and w(z) (e.g. wz_0.83_RP.txt)

- \b VelIniScale \n  factor to rescale v_ini (e.g. 1.10)


*/










/*! \page Gadget-Makefile  Makefile of GADGET-2

There are a number of new types of simulations that can be run with
GADGET-2, most importantly some that use different variants of the
TreePM algorithm. A schematic overview of these simulation-types is
given in Table 1.

The TreePM algorithm is switched on by passing the desired PM
mesh-size at compile time via the makefile to the code. The relevant
parameter is \b PETAPM, see below. Using an explicit force split, the
long-range force is then computed with Fourier techniques, while the
short-range force is done with the tree. Because the tree needs only
be walked locally, a speed-up can arise, particularly for near to
homogeneous particle distributions, but not only restricted to
them. Periodic and non-periodic boundary conditions are
implemented. In the latter case, the code will internally compute FFTs
of size \b 2*PETAPM in order to allow for the required zero-padding.

Pure SPH simulations can also be run in periodic boxes whose
x-dimension is an integer multiple of the dimensions in the y- and
z-directions. This can be useful for hydrodynamic test calculations,
e.g. shock tubes in 3D.
 
Many aspects of these new features are controlled with compile-time
options in the makefile rather than by the parameterfile. This was
done in order to allow the generation of highly optimised binaries by
the compiler, even when the underlying source allows for many
different ways to run the code. Unfortunately, this technique has the
disadvantage that different simulations may require different binaries
of GADGET. If several simulations are run concurrently, there is
hence the danger that a simulation is started/resumed with the `wrong'
binary. Note that while GADGET checks the plausibility of some of
the most important code options, this is not done for all of them. To
minimise the risk of using the wrong code for a simulation, my
recommendation is therefore to produce a separate executable for each
simulation that is run. For example, a good strategy is to make a copy
of the whole code together with its makefile in the output directory
of each simulation run, and then to use this copy to compile the code
and to run the simulation.

The makefile contains a dummy list of all available compile-time code
options, with most of them commented out by default. To activate a
certain feature, the corresponding parameter should be commented in,
and given the desired value, where appropriate.  At the beginning of
the makefile, there is also a brief explanation of each of the
options.

<b>Important Note:</b>Whenever you change one of the makefile options
described below, a full recompilation of the code is necessary. To
guarantee that this is done, you should give the command <b>make
clean</b>, which will erase all object files, followed by <b>make</b>.

\section secmake1 Options that describe the physics of the simulation

- \b PERIODIC \n Set this if you want to have periodic boundary
   conditions.

- \b COOLING \n This enables radiative cooling and heating. It also
   enables an external UV background which is read from a file called
   <em>TREECOOL</em>.

- \b SFR \n This enables star formation using an effective multiphase
   models. This option requires cooling.

- \b METALS \n This activates the tracking of metal enrichment in gas
   and stars. Note that cooling by metal-lines is not yet included in
   the code.

- \b STELLARAGE \n This stores the formation times of each star
   particle, and includes this information in snapshot files.

- \b WINDS \n This activates galactic winds. Requires star formation.

- \b ISOTROPICWINDS \n This makes the wind isotropic. If not set, the
   wind is spawned in an axial way. Requires winds to be activated.

- \b NOGRAVITY \n This switches off gravity. Makes only sense for pure
   SPH simulations in non-expanding space.

- \b LONG_X/Y/Z \n These options can be used together with PERIODIC and
     NOGRAVITY only.  When set, the options define numerical factors that
     can be used to distorts the periodic simulation cube into a
     parallelepiped of arbitrary aspect ratio. This can be useful for
     idealized SPH tests.

- \b TWODIMS \n This effectively switches of one dimension in SPH,
     i.e. the code follows only 2d hydrodynamics in the xy-, yz-, or
     xz-plane. This only works with NOGRAVITY, and if all coordinates of
     the third axis are exactly equal. Can be useful for idealized SPH
     tests.

- \b SPH_BND_PARTICLES \n If this is set, particles with a particle-ID
     equal to zero do not receive any SPH acceleration. This can be useful
     for idealized SPH tests, where these particles represent fixed
     "walls".

\section secmake2 Options that affect SPH

- \b NOFIXEDMASSINKERNEL \n If set, the number of SPH particles in the
   kernel is kept constant instead of the mass contained in the
   kernel.

- \b NOGRADHSML \n If activated, an SPH equation of motion without
   \f$\nabla h\f$-correction factors is used.  Note, for the default
   `entropy'-formulation of SPH (Springel and Hernquist, 2002, MNRAS,
   333, 649), the switches <b>NOFIXEDMASSINKERNEL</b> and
   <b>NOGRADHSML</b> should <em>not</em> be set.

- \b NOVISCOSITYLIMITER \n If this is set, there is no explicit upper
   limit on the viscosity.  In the default version, this limiter will
   try to protect against possible particle `reflections', which may
   in principle occur for poor timestepping in the presence of strong
   shocks.


\section secmake3 Options for the numerical algorithms

- \b PETAPM \n This enables the TreePM method, i.e. the long-range
   force is computed with a PM-algorithm, and the short range force
   with the tree. The parameter has to be set to the size of the mesh
   that should be used, e.g.~64, 96, 128, etc. The mesh dimensions
   need not necessarily be a power of two, but the FFT is fastest for
   such a choice.  Note: If the simulation is not in a periodic box,
   then a FFT method for vacuum boundaries is employed, using a mesh
   with dimension twice that specified by <b>PETAPM</b>.

- \b NOTREERND \n If this is not set, the tree construction will
   succeed even when there are a few particles at identical
   locations. This is done by `rerouting' particles once the node-size
   has fallen below \f$10^{-3}\f$ of the softening length. When this
   option is activated, this will be suppressed and the tree
   construction will always fail if there are particles at extremely
   close or identical coordinates.

- \b NOSTOP_WHEN_BELOW_MINTIMESTEP \n If this is
   activated, the code will not terminate when the timestep falls
   below the value of \b MinSizeTimestep specified in the
   parameterfile. This is useful for runs where one wants to enforce a
   constant timestep for all particles. This can be done by activating
   this option, and by setting \b MinSizeTimestep} and \b
   MaxSizeTimestep to an equal value.


- \b NOPMSTEPADJUSTMENT \n When this is set, the long-range timestep
   for the PM force computation is always determined by \b
   MaxSizeTimeStep.  Otherwise, it is set to the minimum of \b
   MaxSizeTimeStep and the timestep obtained for the maximum
   long-range force with an effective softening scale equal to the PM
   smoothing-scale.


\section secmake4 Architecture specific options

- \b T3E \n The code assumes that \b sizeof(int)=4 holds. A few
   machines (like Cray T3E) have \b sizeof(int)=8. In this case, set
   the T3E flag.


- \b NOTYPEPREFIX_FFTW \n If this is set, the fftw-header/libraries
   are accessed without type prefix (adopting whatever was chosen as
   default at compile-time of fftw). Otherwise, the type prefix 'd'
   for double-precision is used.
 
\section secmake5 Input options

- \b MOREPARAMS \n Activate this to allow a set of additional
   parameters in the parameterfile which control the star formation
   and feedback sector. This option must be activated when star
   formation is switched on.


\section secmake6 Output options

- \b OUTPUTPOTENTIAL \n This will force the code to compute
   gravitational potentials for all particles each time a snapshot
   file is generated. These values are then included in the snapshot
   files. Note that the computation of the values of the potential
   costs additional time.

- \b OUTPUTACCELERATION \n This will include the physical acceleration
   of each particle in snapshot files.

- \b OUTPUTCHANGEOFENTROPY \n This will include the rate of change of
   entropy of gas particles in snapshot files.


- \b OUTPUTTIMESTEP \n This will include the timesteps actually taken
   by each particle in the snapshot files.


\section secmake7 Miscellaneous options

- \b PEANOHILBERT \n This is a tuning option. When set, the code will
   bring the particles into Peano-Hilbert order after each domain
   decomposition. This improves cache utilisation and performance.

- \b WALLCLOCK \n If set, a wallclock timer is used by the code to
   measure internal time consumption (see cpu-log file).  Otherwise, a
   timer that measures consumed processor ticks is used.

- \b FORCETEST=0.01 \n This can be set to check the force accuracy of
   the code, and is only included as a debugging option. The option
   needs to be set to a number between 0 and 1 (e.g. 0.01), which
   specifies the fraction of randomly chosen particles for which at
   each timestep forces by direct summation are computed. The normal
   tree-forces and the `correct' direct summation forces are then
   collected in a file \b forcetest.txt for later inspection. Note
   that the simulation itself is unaffected by this option, but it
   will of course run much(!)  slower, particularly if <b>
   FORCETEST*NumPart*NumPart</b> \f$>>\f$ \b NumPart. Note: Particle IDs
   must be set to numbers \f$>=1\f$ for this option to work.

*/
