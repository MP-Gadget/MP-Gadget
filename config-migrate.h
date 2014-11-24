#ifndef H5_USE_16_API
#error the code requres H5_USE_16_API. define it in your CFLAGS
#endif

#ifdef PETA_PM
#error PETA_PM flag is deprecated use PETAPM and PETAPM_ORDER
#endif
#ifdef PMGRID
#error the PMGRID module is deprecated; use PETAPM.
#endif
#ifdef SWALLOWGAS
#error SWALLOWGAS shall be replaced with BH_SWALLOWGAS also check BH_SWALLOWBH
#endif
#ifdef REPOSITION_ON_POTMIN
#error REPOSITION_ON_POTMIN   shall be replacedwith BH_REPOSITION_ON_POTMIN
#endif
#ifdef BONDI
#error BONDI is replaced with BH_ACCRETION
#endif
#ifdef ENFORCE_EDDINGTON_LIMIT
#error ENFORCE_EDDINGTON_LIMIT is replaced with BH_ENFORCE_EDDINGTON_LIMIT
#endif
#ifdef USE_GASVEL_IN_BONDI
#error USE_GASVEL_IN_BONDI is replaced with BH_USE_GAS_VEL_IN_BONDI
#endif
