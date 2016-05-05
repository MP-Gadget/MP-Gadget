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
    printf("choices %s\n", strchoices);

    for(token = strtok(strchoices, delim); token ; token = strtok(NULL, delim)) {
        printf("token %s\n", token);
        for(p = table; p->name; p++) {
            printf("testing %s\n", p->name);
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
    for(p = table; p->name; p++) {
        if(value & p->value == p->value) {
            strcpy(c, p->name);
            c += strlen(p->name);
            c[0] = '&';
            c++;
            c[0] = 0;
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


static void param_emit(ParameterSet * ps, char * start, int size)
{
    /* parse a line */
    char * buf = alloca(size + 1);
    char * buf1 = alloca(size + 1);
    char * buf2 = alloca(size + 1);
    char * buf3 = alloca(size + 1);

    strncpy(buf, start, size);
    buf[size] = 0;
    if (size == 0) return;
    if (sscanf(buf, "%s%s%s", buf1, buf2, buf3) < 2) return;
    if (buf1[0] == '%') return;
    if (buf1[0] == '#') return;
    /* now this line is important */
    ParameterSchema * p = param_get_schema(ps, buf1);
    if(!p) {
        /* Ignore unknown parameters */
        return;
    }
    param_set_from_string(ps, buf1, buf2);
    printf("%s = %s\n", buf1, buf2);
    if(p->action) {
        p->action(ps, buf1, p->action_data);
    }
}

void param_parse (ParameterSet * ps, char * content)
{
    int i;
    /* copy over the default values */
    for(i = 0; i < ps->size; i ++) {
        ps->value[i] = ps->p[i].defvalue;
    }
    char * p = content;
    char * p1 = content; /* begining of a line */

    while(1) {
        if(*p == '\n' || *p == 0) {
            param_emit(ps, p1, p - p1);
            if(*p == 0) break;
            p++;
            p1 = p;
        } else {
            p++;
        }
    }
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
        p->defvalue.s = strdup(defvalue);
        p->defvalue.nil = 0;
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
            sprintf(buf, "%d", &i);
            return strdup(buf);
        }
        break;
        case DOUBLE:
        {
            int d = ps->value[p->index].d;
            char buf[128];
            sprintf(buf, "%g", &d);
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
    param_declare_string(ps, "InitCondFile", 1, NULL, "Path to the Initial Condition File");
    param_declare_string(ps, "OutputDir",    1, NULL, "Prefix to the output files");
    param_declare_string(ps, "TreeCoolFile", 1, NULL, "Path to the Cooling Table");
    param_declare_string(ps, "MetalCoolFile", 1, NULL, "Path to the Metal Cooling Table");
    param_declare_string(ps, "UVFluctuationFile", 1, NULL, "Path to ");
    param_declare_int(ps,    "DensityKernelType", 1, 0, "");
    param_declare_string(ps, "SnapshotFileBase", 1, NULL, "");
    param_declare_string(ps, "EnergyFile", 0, "energy.txt", "");
    param_declare_string(ps, "CpuFile", 0, "cpu.txt", "");
    param_declare_string(ps, "InfoFile", 0, "info.txt", "");
    param_declare_string(ps, "TimingsFile", 0, "timings.txt", "");
    param_declare_string(ps, "RestartFile", 0, "restart", "");
    param_declare_string(ps, "OutputListFilename", 1, NULL, "");

    param_declare_double(ps, "Omega0", 1, 0.2814, "");
    param_declare_double(ps, "OmegaBaryon", 1, 0.0464, "");
    param_declare_double(ps, "OmegaLambda", 1, 0.7186, "");
    param_declare_double(ps, "HubbleParam", 1, 0.697, "");
    param_declare_double(ps, "BoxSize", 1, 32000, "");

    param_declare_int(ps,    "MaxMemSizePerCore", 0, 1200, "");
    param_declare_double(ps, "TimeOfFirstSnapshot", 0, 0, "");
    param_declare_double(ps, "CpuTimeBetRestartFile", 1, 0, "");
    param_declare_double(ps, "TimeBetStatistics", 0, 0.1, "");
    param_declare_double(ps, "TimeBegin", 1, 0, "");
    param_declare_double(ps, "TimeMax", 0, 1.0, "");
    param_declare_double(ps, "TimeLimitCPU", 1, 0, "");

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

    param_declare_double(ps, "BufferSize", 1, 0, "");
    param_declare_double(ps, "PartAllocFactor", 1, 0, "");
    param_declare_double(ps, "TopNodeAllocFactor", 1, 0, "");

    param_declare_double(ps, "InitGasTemp", 1, 0, "");
    param_declare_double(ps, "MinGasTemp", 1, 0, "");

#if defined(ADAPTIVE_GRAVSOFT_FORGAS) && !defined(ADAPTIVE_GRAVSOFT_FORGAS_HSML)
    param_declare_double(ps, "ReferenceGasMass", 1, 0, "");
#endif

#ifdef FOF
    param_declare_double(ps, "FOFHaloLinkingLength", 1, 0, "");
    param_declare_int(ps, "FOFHaloMinLength", 0, 32, "");
#endif

#ifdef BLACK_HOLES
    param_declare_double(ps, "TimeBetBlackHoleSearch", 1, 0, "");
    param_declare_double(ps, "BlackHoleAccretionFactor", 0, 100, "");
    param_declare_double(ps, "BlackHoleEddingtonFactor", 0, 3, "");
    param_declare_double(ps, "SeedBlackHoleMass", 1, 0, "");
    param_declare_double(ps, "MinFoFMassForNewSeed", 0, 5e-5, "");

    param_declare_double(ps, "BlackHoleNgbFactor", 0, 2, "");

    param_declare_double(ps, "BlackHoleMaxAccretionRadius", 0, 0, "");
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

