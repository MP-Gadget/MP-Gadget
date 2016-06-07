#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "allvars.h"
#include "param.h"

#define INT 1
#define DOUBLE 3
#define STRING 5
#define ENUM 10

static int parse_enum(ParameterEnum * table, char * strchoices) {
    int value = 0;
    ParameterEnum * p = table;
    char * delim = ",;&| \t";
    char * token;

    for(token = strtok(strchoices, delim); token ; token = strtok(NULL, delim)) {
        for(p = table; p->name; p++) {
            if(strcasecmp(token, p->name) == 0) {
                value |= p->value;
                break;
            }
        }
        if(p->name == NULL) {
            /* error occured !*/
            return 0;
        }
    }
    if(value == 0) {
        /* none is specified, use default (NULL named entry) */
        value = p->value;
    }
    return value;
}
static char * format_enum(ParameterEnum * table, int value) {
    char buffer[2048];
    ParameterEnum * p;
    char * c = buffer;
    int first = 1;
    for(p = table; p->name && p->name[0]; p++) {
        if((value & p->value) == p->value) {
            if(!first) {
                *(c++) = ' ';
                *(c++) = '|';
                *(c++) = ' ';
            }
            first = 0;
            strcpy(c, p->name);
            c += strlen(p->name);
            *c = 0;
            value -= p->value;
        }
    }
    return strdup(buffer);
}

typedef struct ParameterValue {
    int nil;
    int i;
    double d;
    char * s;
} ParameterValue;

typedef struct ParameterSchema {
    int index;
    char name[128];
    int type;
    ParameterValue defvalue;
    char * help;
    int required;
    ParameterEnum * enumtable;
    ParameterAction action;
    void * action_data;
} ParameterSchema;

struct ParameterSet {
    char * content;
    int size;
    ParameterSchema p[1024];
    ParameterValue value[1024];
};

static ParameterSchema * param_get_schema(ParameterSet * ps, char * name)
{
    int i;
    for(i = 0; i < ps->size; i ++) {
        if(!strcasecmp(ps->p[i].name, name)) {
            return &ps->p[i];
        }
    }
    return NULL;
}


static int param_emit(ParameterSet * ps, char * start, int size)
{
    /* parse a line */
    char * buf = alloca(size + 1);
    static char blanks[] = " \t\r\n";
    static char comments[] =  "%#";
    strncpy(buf, start, size);
    buf[size] = 0;
    if (size == 0) return 0;

    /* blank lines are OK */
    char * name = NULL;
    char * value = NULL;
    char * ptr = buf;

    /* parse name */
    while(*ptr && strchr(blanks, *ptr)) ptr++;
    if (*ptr == 0 || strchr(comments, *ptr)) {
        /* This line is fully comment */
        return 0;
    }
    name = ptr;
    while(*ptr && !strchr(blanks, *ptr)) ptr++;
    *ptr = 0;
    ptr++;

    /* parse value */
    while(*ptr && strchr(blanks, *ptr)) ptr++;

    if (*ptr == 0 || strchr(comments, *ptr)) {
        /* This line is malformed, must have a value! */
        strncpy(buf, start, size);
        printf("Line `%s` is malformed.\n", buf);
        return 1;
    }
    value = ptr;
    while(*ptr && !strchr(comments, *ptr)) ptr++;
    *ptr = 0;
    ptr++;

    /* now this line is important */
    ParameterSchema * p = param_get_schema(ps, name);
    if(!p) {
        printf("Parameter `%s` is unknown.\n", name);
        return 1;
    }
    param_set_from_string(ps, name, value);
    if(p->action) {
        printf("Triggering Action on `%s`\n", name);
        return p->action(ps, name, p->action_data);
    }
    return 0;
}
int param_validate(ParameterSet * ps)
{
    int i;
    int flag = 0;
    /* copy over the default values */
    for(i = 0; i < ps->size; i ++) {
        ParameterSchema * p = &ps->p[i];
        if(p->required && ps->value[p->index].nil) {
            printf("Parameter `%s` is required, but not set.\n", p->name);
            flag = 1;
        }
    }
    return flag;
}

void param_dump(ParameterSet * ps, FILE * stream)
{
    int i;
    for(i = 0; i < ps->size; i ++) {
        ParameterSchema * p = &ps->p[i];
        char * v = param_format_value(ps, p->name);
        fprintf(stream, "%-31s %s\n", p->name, v);
        free(v);
    }
    fflush(stream);
}

int param_parse (ParameterSet * ps, char * content)
{
    int i;
    /* copy over the default values */
    /* we may want to do ths in get_xxxx, and check for nil. */
    for(i = 0; i < ps->size; i ++) {
        ps->value[ps->p[i].index] = ps->p[i].defvalue;
    }
    char * p = content;
    char * p1 = content; /* begining of a line */
    int flag = 0;
    while(1) {
        if(*p == '\n' || *p == 0) {
            flag |= param_emit(ps, p1, p - p1);
            if(*p == 0) break;
            p++;
            p1 = p;
        } else {
            p++;
        }
    }
    return flag;
}

static ParameterSchema * 
param_declare(ParameterSet * ps, char * name, int type, int required, char * help)
{
    int free = ps->size;
    strcpy(ps->p[free].name, name);
    ps->p[free].required = required;
    ps->p[free].type = type;
    ps->p[free].index = free;
    ps->p[free].defvalue.nil = 1;
    ps->p[free].action = NULL;
    ps->p[free].defvalue.s = NULL;
    if(help)
        ps->p[free].help = strdup(help);
    ps->size ++;
    return &ps->p[free];
}

void
param_declare_int(ParameterSet * ps, char * name, int required, int defvalue, char * help)
{
    ParameterSchema * p = param_declare(ps, name, INT, required, help);
    if(!required) {
        p->defvalue.i = defvalue;
        p->defvalue.nil = 0;
    }
}
void
param_declare_double(ParameterSet * ps, char * name, int required, double defvalue, char * help)
{
    ParameterSchema * p = param_declare(ps, name, DOUBLE, required, help);
    if(!required) {
        p->defvalue.d = defvalue;
        p->defvalue.nil = 0;
    }
}
void
param_declare_string(ParameterSet * ps, char * name, int required, char * defvalue, char * help)
{
    ParameterSchema * p = param_declare(ps, name, STRING, required, help);
    if(!required) {
        if(defvalue != NULL) {
            p->defvalue.s = strdup(defvalue);
            p->defvalue.nil = 0;
        } else {
            /* The handling of nil is not consistent yet! Only string can be non-required and have nil value.
             * blame bad function signature (noway to define nil for int and double. */
            p->defvalue.s = NULL;
            p->defvalue.nil = 1;
        }
    }
}
void
param_declare_enum(ParameterSet * ps, char * name, ParameterEnum * enumtable, int required, int defvalue, char * help)
{
    ParameterSchema * p = param_declare(ps, name, ENUM, required, help);
    p->enumtable = enumtable;
    if(!required) {
        p->defvalue.i = defvalue;
        /* Watch out, if enumtable is malloced we may core dump if it gets freed */
        p->defvalue.nil = 0;
    }
}

void
param_set_action(ParameterSet * ps, char * name, ParameterAction action, void * userdata)
{
    ParameterSchema * p = param_get_schema(ps, name);
    p->action = action;
    p->action_data = userdata;
}

int
param_is_nil(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].nil;
}

double
param_get_double(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].d;
}

char *
param_get_string(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].s;
}
void
param_get_string2(ParameterSet * ps, char * name, char * dst)
{
    ParameterSchema * p = param_get_schema(ps, name);
    strcpy(dst, ps->value[p->index].s);
}

int
param_get_int(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].i;
}

int
param_get_enum(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].i;
}

char *
param_format_value(ParameterSet * ps, char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if(ps->value[p->index].nil) {
        return strdup("NIL");
    }
    switch(p->type) {
        case INT:
        {
            int i = ps->value[p->index].i;
            char buf[128];
            sprintf(buf, "%d", i);
            return strdup(buf);
        }
        break;
        case DOUBLE:
        {
            double d = ps->value[p->index].d;
            char buf[128];
            sprintf(buf, "%g", d);
            return strdup(buf);
        }
        break;
        case STRING:
        {
            return strdup(ps->value[p->index].s);
        }
        break;
        case ENUM:
        {
            return format_enum(p->enumtable, ps->value[p->index].i);
        }
        break;
    }
    return NULL;
}

void
param_set_from_string(ParameterSet * ps, char * name, char * value)
{
    ParameterSchema * p = param_get_schema(ps, name);
    switch(p->type) {
        case INT:
        {
            int i;
            sscanf(value, "%d", &i);
            ps->value[p->index].i = i;
            ps->value[p->index].nil = 0;
        }
        break;
        case DOUBLE:
        {
            double d;
            sscanf(value, "%lf", &d);
            ps->value[p->index].d = d;
            ps->value[p->index].nil = 0;
        }
        break;
        case STRING:
        {
            ps->value[p->index].s = strdup(value);
            ps->value[p->index].nil = 0;
        }
        break;
        case ENUM:
        {
            char * v = strdup(value);
            ps->value[p->index].i = parse_enum(p->enumtable, v);
            free(v);
            ps->value[p->index].nil = 0;
        }
        break;
    }
}

ParameterSet *
parameter_set_new()
{
    ParameterSet * ps = malloc(sizeof(ParameterSet));
    ps->size = 0;
    param_declare_string(ps, "InitCondFile", 1, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    1, NULL, "Prefix to the output files");
    param_declare_string(ps, "TreeCoolFile", 1, NULL, "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", 0, "", "Path to the Metal Cooling Table. Refer to cooling.c");
    param_declare_string(ps, "UVFluctuationFile", 0, "", "Path to the UVFluctation Table. Refer to cooling.c.");

    static ParameterEnum DensityKernelTypeEnum [] = {
        {"cubic", DENSITY_KERNEL_CUBIC_SPLINE},
        {"quintic", DENSITY_KERNEL_QUINTIC_SPLINE},
        {"quartic", DENSITY_KERNEL_QUARTIC_SPLINE},
        {NULL, DENSITY_KERNEL_QUARTIC_SPLINE},
    } ;
    param_declare_enum(ps,    "DensityKernelType", DensityKernelTypeEnum, 1, 0, "");
    param_declare_string(ps, "SnapshotFileBase", 1, NULL, "");
    param_declare_string(ps, "EnergyFile", 0, "energy.txt", "");
    param_declare_string(ps, "CpuFile", 0, "cpu.txt", "");
    param_declare_string(ps, "InfoFile", 0, "info.txt", "");
    param_declare_string(ps, "TimingsFile", 0, "timings.txt", "");
    param_declare_string(ps, "RestartFile", 0, "restart", "");
    param_declare_string(ps, "OutputList", 1, NULL, "List of output times");

    param_declare_double(ps, "Omega0", 1, 0.2814, "");
    param_declare_double(ps, "CMBTemperature", 0, 2.7255,
            "Present-day CMB temperature in Kelvin, default from Fixsen 2009; affects background if RadiationOn is set.");
    param_declare_double(ps, "OmegaBaryon", 1, 0.0464, "");
    param_declare_double(ps, "OmegaLambda", 1, 0.7186, "");
    param_declare_double(ps, "HubbleParam", 1, 0.697, "");
    param_declare_double(ps, "BoxSize", 1, 32000, "");

    param_declare_int(ps,    "MaxMemSizePerCore", 0, 1200, "");
    param_declare_double(ps, "CpuTimeBetRestartFile", 1, 0, "");

    param_declare_double(ps, "TimeBegin", 1, 0, "");
    param_declare_double(ps, "TimeMax", 0, 1.0, "");
    param_declare_double(ps, "TimeLimitCPU", 1, 0, "");

    param_declare_int   (ps, "DomainOverDecompositionFactor", 0, 1, "Number of sub domains on a MPI rank");
    param_declare_double(ps, "TreeDomainUpdateFrequency", 0, 0.025, "");
    param_declare_double(ps, "ErrTolTheta", 0, 0.5, "");
    param_declare_int(ps,    "TypeOfOpeningCriterion", 0, 1, "");
    param_declare_double(ps, "ErrTolIntAccuracy", 0, 0.02, "");
    param_declare_double(ps, "ErrTolForceAcc", 0, 0.005, "");
    param_declare_int(ps,    "Nmesh", 1, 0, "");

    param_declare_double(ps, "MinGasHsmlFractional", 0, 0, "");
    param_declare_double(ps, "MaxGasVel", 0, 3e5, "");

    param_declare_int(ps,    "TypeOfTimestepCriterion", 0, 0, "Magic numbers!");
    param_declare_double(ps, "MaxSizeTimestep", 0, 0.1, "");
    param_declare_double(ps, "MinSizeTimestep", 0, 0, "");

    param_declare_double(ps, "MaxRMSDisplacementFac", 0, 0.2, "");
    param_declare_double(ps, "ArtBulkViscConst", 0, 0.75, "");
    param_declare_double(ps, "CourantFac", 0, 0.15, "");
    param_declare_double(ps, "DensityResolutionEta", 0, 1.0, "Resolution eta factor (See Price 2008) 1 = 33 for Cubic Spline");

    param_declare_double(ps, "DensityContrastLimit", 0, 100, "Max contrast for hydro force calculation");
    param_declare_double(ps, "MaxNumNgbDeviation", 0, 2, "");

    param_declare_int(ps, "NumFilesPerSnapshot", 1, 0, "");
    param_declare_int(ps, "NumWritersPerSnapshot", 1, 0, "");
    param_declare_int(ps, "NumFilesPerPIG", 1, 0, "");
    param_declare_int(ps, "NumWritersPerPIG", 1, 0, "");

    param_declare_int(ps, "CoolingOn", 1, 0, "");
    param_declare_int(ps, "StarformationOn", 1, 0, "");
    param_declare_int(ps, "RadiationOn", 0, 0, "Include radiation density in the background evolution.");
    param_declare_int(ps, "FastParticleType", 0, 2, "Particles of this type will not decrease the timestep. Default neutrinos.");
    param_declare_int(ps, "NoTreeType", 0, 2, "Particles of this type will not produce tree forces. Default neutrinos.");

    param_declare_double(ps, "SofteningHalo", 1, 0, "");
    param_declare_double(ps, "SofteningDisk", 1, 0, "");
    param_declare_double(ps, "SofteningBulge", 1, 0, "");
    param_declare_double(ps, "SofteningGas", 1, 0, "");
    param_declare_double(ps, "SofteningStars", 1, 0, "");
    param_declare_double(ps, "SofteningBndry", 1, 0, "");
    param_declare_double(ps, "SofteningHaloMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningDiskMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningBulgeMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningGasMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningStarsMaxPhys", 1, 0, "");
    param_declare_double(ps, "SofteningBndryMaxPhys", 1, 0, "");

    param_declare_double(ps, "BufferSize", 0, 100, "");
    param_declare_double(ps, "PartAllocFactor", 1, 0, "");
    param_declare_double(ps, "TopNodeAllocFactor", 0, 0.5, "");

    param_declare_double(ps, "InitGasTemp", 1, 0, "");
    param_declare_double(ps, "MinGasTemp", 1, 0, "");

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
    param_declare_double(ps, "ReferenceGasMass", 1, 0, "");
#endif

#ifdef FOF
    param_declare_double(ps, "FOFHaloLinkingLength", 1, 0, "");
    param_declare_int(ps, "FOFHaloMinLength", 0, 32, "");
    param_declare_double(ps, "MinFoFMassForNewSeed", 0, 5e2, "Minimal Mass for seeding tracer particles ");
    param_declare_double(ps, "TimeBetweenSeedingSearch", 0, 1e5, "Time Between Seeding Attempts: default to a a large value, meaning never.");
#endif

#ifdef BLACK_HOLES
    param_declare_double(ps, "BlackHoleAccretionFactor", 0, 100, "");
    param_declare_double(ps, "BlackHoleEddingtonFactor", 0, 3, "");
    param_declare_double(ps, "SeedBlackHoleMass", 1, 0, "");

    param_declare_double(ps, "BlackHoleNgbFactor", 0, 2, "");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", 0, 99999., "");
    param_declare_double(ps, "BlackHoleFeedbackFactor", 0, 0.05, "");
    param_declare_double(ps, "BlackHoleFeedbackRadius", 1, 0, "");

    param_declare_double(ps, "BlackHoleFeedbackRadiusMaxPhys", 1, 0, "");

    static ParameterEnum BlackHoleFeedbackMethodEnum [] = {
        {"mass", BH_FEEDBACK_MASS},
        {"volume", BH_FEEDBACK_VOLUME},
        {"tophat", BH_FEEDBACK_TOPHAT},
        {"spline", BH_FEEDBACK_SPLINE},
        {NULL, BH_FEEDBACK_SPLINE | BH_FEEDBACK_MASS},
    };
    param_declare_enum(ps, "BlackHoleFeedbackMethod", BlackHoleFeedbackMethodEnum, 1, 0, "");
#endif

#ifdef SFR
    static ParameterEnum StarformationCriterionEnum [] = {
        {"density", SFR_CRITERION_DENSITY},
        {"h2", SFR_CRITERION_MOLECULAR_H2},
        {"selfgravity", SFR_CRITERION_SELFGRAVITY},
        {"convergent", SFR_CRITERION_CONVERGENT_FLOW},
        {"continous", SFR_CRITERION_CONTINUOUS_CUTOFF},
        {NULL, SFR_CRITERION_DENSITY},
    };

    static ParameterEnum WindModelEnum [] = {
        {"subgrid", WINDS_SUBGRID},
        {"decouple", WINDS_DECOUPLE_SPH},
        {"halo", WINDS_USE_HALO},
        {"fixedefficiency", WINDS_FIXED_EFFICIENCY},
        {"sh03", WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY} ,
        {"vs08", WINDS_FIXED_EFFICIENCY},
        {"ofjt10", WINDS_USE_HALO | WINDS_DECOUPLE_SPH},
        {"isotropic", WINDS_ISOTROPIC },
        {"nowind", WINDS_NONE},
        {NULL, WINDS_SUBGRID | WINDS_DECOUPLE_SPH | WINDS_FIXED_EFFICIENCY},
    };

    param_declare_enum(ps, "StarformationCriterion", StarformationCriterionEnum, 1, 0, "");

    param_declare_double(ps, "CritOverDensity", 0, 57.7, "");
    param_declare_double(ps, "CritPhysDensity", 0, 0, "");

    param_declare_double(ps, "FactorSN", 0, 0.1, "");
    param_declare_double(ps, "FactorEVP", 0, 1000, "");
    param_declare_double(ps, "TempSupernova", 0, 1e8, "");
    param_declare_double(ps, "TempClouds", 0, 1000, "");
    param_declare_double(ps, "MaxSfrTimescale", 0, 1.5, "");
    param_declare_enum(ps, "WindModel", WindModelEnum, 1, 0, "");

    /* The following two are for VS08 and SH03*/
    param_declare_double(ps, "WindEfficiency", 0, 2.0, "");
    param_declare_double(ps, "WindEnergyFraction", 0, 1.0, "");

    /* The following two are for OFJT10*/
    param_declare_double(ps, "WindSigma0", 0, 353, "");
    param_declare_double(ps, "WindSpeedFactor", 0, 3.7, "");

    param_declare_double(ps, "WindFreeTravelLength", 0, 20, "");
    param_declare_double(ps, "WindFreeTravelDensFac", 0, 0., "");

    param_declare_double(ps, "QuickLymanAlphaProbability", 0, 0, "");

#endif

#ifdef SOFTEREQS
    param_declare_double(ps, "FactorForSofterEQS", 1, 0, "");
#endif
    return ps;
}
void
parameter_set_free(ParameterSet * ps) {
    int i;
    for(i = 0; i < ps->size; i ++) {
        if(ps->p[i].help) {
            free(ps->p[i].help);
        }
        if(ps->p[i].type == STRING) {
            if(ps->p[i].defvalue.s) {
//                free(ps->p[i].defvalue.s);
            }
            if(ps->value[ps->p[i].index].s != ps->p[i].defvalue.s) {
                if(ps->value[ps->p[i].index].s) {
//                    free(ps->value[ps->p[i].index].s);
                }
            }
        }
    }
    free(ps);
}

