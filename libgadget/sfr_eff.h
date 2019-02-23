#ifndef __SFR_H
#define __SFR_H

#ifndef  GENERATIONS
#define  GENERATIONS     4	/*!< Number of star particles that may be created per gas particle */
#endif

#define  METAL_YIELD       0.02	/*!< effective metal yield for star formation */

void init_cooling_and_star_formation(void);
void cooling_and_starformation(void);
double get_starformation_rate(int i);

#endif
