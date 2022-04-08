#include "unitsystem.h"
#include <math.h>

/* Construct a unit system struct*/
struct UnitSystem
get_unitsystem(double UnitLength_in_cm, double UnitMass_in_g, double UnitVelocity_in_cm_per_s)
{
    struct UnitSystem units;
    units.UnitMass_in_g = UnitMass_in_g;
    units.UnitVelocity_in_cm_per_s = UnitVelocity_in_cm_per_s;
    units.UnitLength_in_cm = UnitLength_in_cm;

    units.UnitTime_in_s = units.UnitLength_in_cm / units.UnitVelocity_in_cm_per_s;
    units.UnitDensity_in_cgs = units.UnitMass_in_g / pow(units.UnitLength_in_cm, 3);
    units.UnitEnergy_in_cgs = units.UnitMass_in_g * pow(units.UnitLength_in_cm, 2) / pow(units.UnitTime_in_s, 2);
    units.UnitInternalEnergy_in_cgs = units.UnitEnergy_in_cgs / units.UnitMass_in_g;
    return units;
}
