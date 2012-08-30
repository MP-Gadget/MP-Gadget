/*! \file assert.h
 *  \brief definition of an assertation macro that
 *  behaves more coherent with the Gadget-Framework
 *  than the standard definition of the macro
 */

#ifndef DEBUG

#define assert(expr)

#else

#ifdef __func__
#define assert(expr) if (!(expr)) { printf("ASSERTATION FAULT: File: %s -- %s() -- Line: %i -- Assertation (%s) failed.\n", __FILE__, __func__, __LINE__, __STRING(expr) ); fflush(stdout); endrun(1337); }
#else
#define assert(expr) if (!(expr)) { printf("ASSERTATION FAULT: File: %s -- Line: %i -- Assertation (%s) failed.\n", __FILE__, __LINE__, __STRING(expr) ); fflush(stdout); endrun(1337); }
#endif
#endif 
