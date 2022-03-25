#ifndef UNIT_SYSTEM_H
#define UNIT_SYSTEM_H

/* System of units:
 * Convert from internal units into normal physical units.
 * The factors of h are left in.*/
struct UnitSystem {
    double UnitMass_in_g;		/*!< factor to convert internal mass unit to grams/h */
    double UnitVelocity_in_cm_per_s;	/*!< factor to convert intqernal velocity unit to cm/sec */
    double UnitLength_in_cm;		/*!< factor to convert internal length unit to cm/h */
    double UnitTime_in_s,		/*!< factor to convert internal time unit to seconds/h */
           UnitDensity_in_cgs,		/*!< factor to convert internal length unit to g/cm^3*h^2 */
           UnitEnergy_in_cgs;	/*!< factor to convert internal energy unit to cgs units */
    double UnitInternalEnergy_in_cgs;  /*< factor to convert the internal unit of internal energy to cgs units */
};

/* Construct a unit system struct*/
struct UnitSystem get_unitsystem(const double UnitLength_in_cm, const double UnitMass_in_g, const double UnitVelocity_in_cm_per_s);

#endif
