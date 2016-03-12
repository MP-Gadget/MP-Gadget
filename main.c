#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>
#include <gsl/gsl_math.h>

#include <fftw3-mpi.h>

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
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &ThisTask);
  MPI_Comm_size(MPI_COMM_WORLD, &NTask);

  fftw_init_threads();
  fftw_mpi_init();
  fftw_plan_with_nthreads(omp_get_max_threads());

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
	  printf("       3          Run FOF if enabled\n");
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

Note that PFFT needs to be compiled with parallel support enabled.
This can be achieved by passing the option <b>--enable-mpi</b> to the
configure script.

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

- \b NOVISCOSITYLIMITER \n If this is set, there is no explicit upper
   limit on the viscosity.  In the default version, this limiter will
   try to protect against possible particle `reflections', which may
   in principle occur for poor timestepping in the presence of strong
   shocks.


\section secmake3 Options for the numerical algorithms

- \b NOTREERND \n If this is not set, the tree construction will
   succeed even when there are a few particles at identical
   locations. This is done by `rerouting' particles once the node-size
   has fallen below \f$10^{-3}\f$ of the softening length. When this
   option is activated, this will be suppressed and the tree
   construction will always fail if there are particles at extremely
   close or identical coordinates.

\section secmake4 Architecture specific options

- \b T3E \n The code assumes that \b sizeof(int)=4 holds. A few
   machines (like Cray T3E) have \b sizeof(int)=8. In this case, set
   the T3E flag.

\section secmake7 Miscellaneous options

- \b PEANOHILBERT \n This is a tuning option. When set, the code will
   bring the particles into Peano-Hilbert order after each domain
   decomposition. This improves cache utilisation and performance.

- \b WALLCLOCK \n If set, a wallclock timer is used by the code to
   measure internal time consumption (see cpu-log file).  Otherwise, a
   timer that measures consumed processor ticks is used.

*/
