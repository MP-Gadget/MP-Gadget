#ifndef PARAMSET_H
#define PARAMSET_H

#include <stdio.h>

enum ParameterFlag {
    REQUIRED = 0,
    OPTIONAL = 1,
    OPTIONAL_UNDEF = 2, /* optional and the default is undefined param_is_nil() is true if no value is given. */
};

typedef struct ParameterEnum {
    char * name;
    int value;
} ParameterEnum;

typedef struct ParameterSet ParameterSet;
typedef int (*ParameterAction)(ParameterSet * ps, char * name, void * userdata);

void
param_declare_int(ParameterSet * ps, char * name, enum ParameterFlag required, int defvalue, char * help);

void
param_declare_double(ParameterSet * ps, char * name, enum ParameterFlag required, double defvalue, char * help);

void
param_declare_string(ParameterSet * ps, char * name, enum ParameterFlag required, char * defvalue, char * help);

void
param_declare_enum(ParameterSet * ps, char * name, ParameterEnum * enumtable, enum ParameterFlag required, char * defvalue, char * help);

void
param_set_action(ParameterSet * ps, char * name, ParameterAction action, void * userdata);

int
param_is_nil(ParameterSet * ps, char * name);

double
param_get_double(ParameterSet * ps, char * name);

char *
param_get_string(ParameterSet * ps, char * name);

void
param_get_string2(ParameterSet * ps, char * name, char * dest, size_t len);
int
param_get_int(ParameterSet * ps, char * name);

int
param_get_enum(ParameterSet * ps, char * name);

char *
param_format_value(ParameterSet * ps, char * name);

int param_parse (ParameterSet * ps, char * content);
/* returns 0 on no error; 1 on error */
int param_parse_file (ParameterSet * ps, const char * filename);
/* returns 0 on no error; 1 on error */
int param_validate(ParameterSet * ps);
void param_dump(ParameterSet * ps, FILE * stream);

ParameterSet *
parameter_set_new();

void
parameter_set_free(ParameterSet * ps);

#endif
