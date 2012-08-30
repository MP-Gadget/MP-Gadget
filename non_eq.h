c
c Written by S. Glover, AMNH, 2004-2005, AIP, 2006-2007
c
#ifdef CHEMCOOL

#ifdef CHEMISTRYNETWORK
      integer nchem_network
      parameter (nchem_network = CHEMISTRYNETWORK)
#endif
c
c Set up quantities (such as the absolute tolerances) that are used in 
c multiple places in the non-equilibrium chemistry code. Note that most 
c DVODE-specific setup should go in evolve_abundances.F -- nrpar & nipar
c are exceptions, as they are used elsewhere, so it is useful to define
c them here
c 
      integer nrpar, nipar
      parameter (nrpar = NRPAR)
      parameter (nipar = NIPAR)
c
      integer num_non_eq_species
      parameter (num_non_eq_species = TRAC_NUM)

      integer num_eqb_species
#if CHEMISTRYNETWORK == 1 || CHEMISTRYNETWORK == 2
      parameter (num_eqb_species = 2)
#endif
#if CHEMISTRYNETWORK == 3
      parameter (num_eqb_species = 14)
#endif
#if CHEMISTRYNETWORK == 4 || CHEMISTRYNETWORK == 5 || CHEMISTRYNETWORK == 6 || CHEMISTRYNETWORK == 8
      parameter (num_eqb_species = 0)
#endif
#if CHEMISTRYNETWORK == 7 || CHEMISTRYNETWORK == 11
      parameter (num_eqb_species = 13)
#endif
#if CHEMISTRYNETWORK == 9
      parameter (num_eqb_species = 6)
#endif
#if CHEMISTRYNETWORK == 10
      parameter (num_eqb_species = 0)
#endif
c
      integer nspec
      parameter (nspec = NSPEC)
c
c Amount by which abundances are allowed to stray over their theoretical
c maximum before triggering an error in rate_eq -- set to a blanket value
c of 1d-4 for the time being...
c 
      REAL eps_max
      parameter (eps_max = EPS_MAX)
c
      REAL atol(nspec)
c
      common /tolerance/ atol
c
#endif /* CHEMCOOL */
