typedef struct ParameterEnum {
    char * name;
    int value;
} ParameterEnum;

typedef struct ParameterSet ParameterSet;
typedef int (*ParameterAction)(ParameterSet * ps, char * name, void * userdata);

void
param_declare_int(ParameterSet * ps, char * name, int required, int defvalue, char * help);

void
param_declare_double(ParameterSet * ps, char * name, int required, double defvalue, char * help);

void
param_declare_string(ParameterSet * ps, char * name, int required, char * defvalue, char * help);

void
param_declare_enum(ParameterSet * ps, char * name, ParameterEnum * enumtable, int required, char * defvalue, char * help);

void
param_set_action(ParameterSet * ps, char * name, ParameterAction action, void * userdata);

int
param_is_nil(ParameterSet * ps, char * name);

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

int param_parse (ParameterSet * ps, char * content);
int param_parse_file (ParameterSet * ps, const char * filename);
int param_validate(ParameterSet * ps); /* 0 for good, 1 for bad; prints messages. */
void param_dump(ParameterSet * ps, FILE * stream);

ParameterSet *
parameter_set_new();

void
parameter_set_free(ParameterSet * ps);

