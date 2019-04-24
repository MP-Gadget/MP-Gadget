#ifndef __SFR_H
#define __SFR_H

#include "forcetree.h"

#ifndef  GENERATIONS
#define  GENERATIONS     4	/*!< Number of star particles that may be created per gas particle */
#endif

#define  METAL_YIELD       0.02	/*!< effective metal yield for star formation */

void init_cooling_and_star_formation(void);
/*Do the cooling and the star formation. The tree is required for the winds only.*/
void cooling_and_starformation(ForceTree * tree);
double get_starformation_rate(int i);

/*Get the neutral fraction of a particle correctly, even when on the star-forming equation of state.
 * This calls the cooling routines for the current internal energy when off the equation of state, but
 * when on the equation of state calls them separately for the cold and hot gas.*/
double get_neutral_fraction_sfreff(int i, double redshift);

#endif
