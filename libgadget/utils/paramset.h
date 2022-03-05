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
typedef int (*ParameterAction)(ParameterSet * ps, const char * name, void * userdata);

void
param_declare_int(ParameterSet * ps, const char * name, const enum ParameterFlag required, const int defvalue, const char * help);

void
param_declare_double(ParameterSet * ps, const char * name, const enum ParameterFlag required, const double defvalue, const char * help);

void
param_declare_string(ParameterSet * ps, const char * name, const enum ParameterFlag required, const char * defvalue, const char * help);

void
param_declare_enum(ParameterSet * ps, const char * name, ParameterEnum * enumtable, const enum ParameterFlag required, const char * defvalue, const char * help);

void
param_set_action(ParameterSet * ps, const char * name, ParameterAction action, void * userdata);

int
param_is_nil(ParameterSet * ps, const char * name);

double
param_get_double(ParameterSet * ps, const char * name);

char *
param_get_string(ParameterSet * ps, const char * name);

void
param_get_string2(ParameterSet * ps, const char * name, char * dest, const size_t len);
int
param_get_int(ParameterSet * ps, const char * name);

int
param_get_enum(ParameterSet * ps, const char * name);

char *
param_format_value(ParameterSet * ps, const char * name);

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
