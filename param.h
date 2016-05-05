
typedef struct ParameterEnum {
    char name[128];
    int value;
} ParameterEnum;

typedef struct ParameterSet ParameterSet;
typedef void (*ParameterAction)(ParameterSet * ps, char * name, void * userdata);

void
param_declare_int(ParameterSet * ps, char * name, int required, int defvalue, char * help);

void
param_declare_double(ParameterSet * ps, char * name, int required, double defvalue, char * help);

void
param_declare_string(ParameterSet * ps, char * name, int required, char * defvalue, char * help);

void
param_declare_enum(ParameterSet * ps, char * name, ParameterEnum * enumtable, int required, int defvalue, char * help);

void
param_set_action(ParameterSet * ps, char * name, ParameterAction action, void * userdata);

double
param_get_double(ParameterSet * ps, char * name);

char *
param_get_string(ParameterSet * ps, char * name);

void
param_get_string2(ParameterSet * ps, char * name, char * dest);
int
param_get_int(ParameterSet * ps, char * name);

int
param_get_enum(ParameterSet * ps, char * name);

char *
param_format_value(ParameterSet * ps, char * name);

void
param_set_from_string(ParameterSet * ps, char * name, char * value);

void param_parse (ParameterSet * ps, char * content);

ParameterSet *
parameter_set_new();

void
parameter_set_free(ParameterSet * ps);

