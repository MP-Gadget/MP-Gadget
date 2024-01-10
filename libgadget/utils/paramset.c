#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "paramset.h"
#include "string.h"
#include "mymalloc.h"
#include "endrun.h"

#define INT 1
#define DOUBLE 3
#define STRING 5
#define ENUM 10
#define NAMESIZE 128

static int parse_enum(ParameterEnum * table, const char * strchoices) {
    int value = 0;
    ParameterEnum * p = table;
    const char * delim = "\",;&| \t";
    char * token;

    char * strchoices2 = fastpm_strdup(strchoices);
    for(token = strtok(strchoices2, delim); token ; token = strtok(NULL, delim)) {
        for(p = table; p->name; p++) {
            if(strcasecmp(token, p->name) == 0) {
                value |= p->value;
                break;
            }
        }
        if(p->name == NULL) {
            /* error occured !*/
            myfree(strchoices2);
            return 0;
        }
    }
    if(value == 0) {
        /* none is specified, use default (NULL named entry) */
        value = p->value;
    }
    myfree(strchoices2);
    return value;
}
static char * format_enum(ParameterEnum * table, int value) {
    int btotal = 200;
    int bleft = btotal;
    char * buffer = ta_malloc("formatbuffer", char, bleft);
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
            bleft-= strlen(p->name);
            if (bleft <= 0) {
                int extra = (- bleft) + btotal;
                buffer = (char *) myrealloc(buffer, btotal + extra);
                break;
            }
            /* The above ensures a large enough buffer*/
            strcpy(c, p->name);
            c += strlen(p->name);
        }
    }
    return buffer;
}

typedef struct ParameterValue {
    int nil;
    int i;
    double d;
    char * s;
    int lineno;
} ParameterValue;

typedef struct ParameterSchema {
    int index;
    char name[NAMESIZE];
    int type;
    ParameterValue defvalue;
    const char * help;
    enum ParameterFlag required;
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

static ParameterSchema * param_get_schema(ParameterSet * ps, const char * name)
{
    int i;
    for(i = 0; i < ps->size; i ++) {
        if(!strcasecmp(ps->p[i].name, name)) {
            return &ps->p[i];
        }
    }
    return NULL;
}

static void
param_set_from_string(ParameterSet * ps, const char * name, char * value, int lineno);


static int param_emit(ParameterSet * ps, char * start, int size, int lineno)
{
    /* parse a line */
    if (size == 0) return 0;
    char * buf = ta_malloc2("line", char, size + 1);
    static char blanks[] = " \t\r\n=";
    static char comments[] =  "%#";
    strncpy(buf, start, size);
    buf[size] = 0;

    /* blank lines are OK */
    const char * name = NULL;
    char * value = NULL;
    char * ptr = buf;

    /* parse name */
    while(*ptr && strchr(blanks, *ptr)) ptr++;
    if (*ptr == 0 || strchr(comments, *ptr)) {
        /* This line is fully comment */
        myfree(buf);
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
        message(0, "Line %d : `%s` is malformed.\n", lineno, buf);
        myfree(buf);
        return 1;
    }
    value = ptr;
    while(*ptr && !strchr(comments, *ptr)) ptr++;
    *ptr = 0;
    ptr++;

    /* now this line is important */
    ParameterSchema * p = param_get_schema(ps, name);
    if(!p) {
        message(0, "Line %d: Parameter `%s` is unknown.\n", lineno, name);
        myfree(buf);
        return 1;
    }
    param_set_from_string(ps, name, value, lineno);
    if(p->action) {
        if(0 != p->action(ps, name, p->action_data)) {
            message(0, "Triggering Action on `%s` failed.\n", name);
            myfree(buf);
            return 1;
        }
    }
    myfree(buf);
    return 0;
}
int param_validate(ParameterSet * ps)
{
    int i;
    int flag = 0;
    /* copy over the default values */
    for(i = 0; i < ps->size; i ++) {
        ParameterSchema * p = &ps->p[i];
        if(p->required == REQUIRED && ps->value[p->index].nil) {
            message(0, "Parameter `%s` is required, but not set.\n", p->name);
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
        if(ps->value[i].lineno >= 0) {
            fprintf(stream, "%-31s %-20s # Line %03d # %s \n", p->name, v, ps->value[i].lineno, p->help);
        } else {
            fprintf(stream, "%-31s %-20s # Default  # %s \n", p->name, v, p->help);
        }
        myfree(v);
    }
    fflush(stream);
}

int param_parse (ParameterSet * ps, char * content)
{
    int i;
    /* copy over the default values, include nil values */
    for(i = 0; i < ps->size; i ++) {
        ps->value[ps->p[i].index] = ps->p[i].defvalue;
    }
    char * p = content;
    char * p1 = content; /* begining of a line */
    int flag = 0;
    int lineno = 0;
    while(1) {
        if(*p == '\n' || *p == 0) {
            int flag1 = param_emit(ps, p1, p - p1, lineno);
            flag |= flag1;
            if(*p == 0) break;
            p++;
            p1 = p;
            lineno ++;
        } else {
            p++;
        }
    }
    return flag;
}

int param_parse_file (ParameterSet * ps, const char * filename)
{
    char * content = fastpm_file_get_content(filename);
    if(content == NULL) {
        endrun(1, "Could not read file: %s\n", filename);
    }
    int val = param_parse(ps, content);

    myfree(content);
    return val;
}

static ParameterSchema *
param_declare(ParameterSet * ps, const char * name, const int type, const enum ParameterFlag required, const char * help)
{
    int free = ps->size;
    strncpy(ps->p[free].name, name, NAMESIZE);
    //Make sure null terminated
    ps->p[free].name[NAMESIZE-1] = '\0';
    ps->p[free].required = required;
    ps->p[free].type = type;
    ps->p[free].index = free;
    ps->p[free].defvalue.nil = 1;
    ps->p[free].action = NULL;
    ps->p[free].defvalue.s = NULL;
    ps->p[free].defvalue.lineno = -1;
    if(help)
        ps->p[free].help = help;
    ps->size ++;
    return &ps->p[free];
}

void
param_declare_int(ParameterSet * ps, const char * name, const enum ParameterFlag required, const int defvalue, const char * help)
{
    ParameterSchema * p = param_declare(ps, name, INT, required, help);
    if(required == OPTIONAL) {
        p->defvalue.i = defvalue;
        p->defvalue.nil = 0;
    } else {
        p->defvalue.nil = 1;
    }
}
void
param_declare_double(ParameterSet * ps, const char * name, const enum ParameterFlag required, const double defvalue, const char * help)
{
    ParameterSchema * p = param_declare(ps, name, DOUBLE, required, help);
    if(required == OPTIONAL) {
        p->defvalue.d = defvalue;
        p->defvalue.nil = 0;
    } else {
        p->defvalue.nil = 1;
    }
}
void
param_declare_string(ParameterSet * ps, const char * name, const enum ParameterFlag required, const char * defvalue, const char * help)
{
    ParameterSchema * p = param_declare(ps, name, STRING, required, help);
    if(required == OPTIONAL) {
        if(defvalue != NULL) {
            p->defvalue.s = fastpm_strdup(defvalue);
            p->defvalue.nil = 0;
        } else {
            /* The handling of nil is not consistent yet! Only string can be non-required and have nil value.
             * blame bad function signature (noway to define nil for int and double. */
            p->defvalue.s = NULL;
            p->defvalue.nil = 1;
        }
    } else {
        p->defvalue.nil = 1;
    }
}
void
param_declare_enum(ParameterSet * ps, const char * name, ParameterEnum * enumtable, const enum ParameterFlag required, const char * defvalue, const char * help)
{
    ParameterSchema * p = param_declare(ps, name, ENUM, required, help);
    p->enumtable = enumtable;
    if(required == OPTIONAL) {
        p->defvalue.i = parse_enum(enumtable, defvalue);
        /* Watch out, if enumtable is malloced we may core dump if it gets freed */
        p->defvalue.nil = 0;
    } else {
        p->defvalue.nil = 1;
    }
}

void
param_set_action(ParameterSet * ps, const char * name, ParameterAction action, void * userdata)
{
    ParameterSchema * p = param_get_schema(ps, name);
    p->action = action;
    p->action_data = userdata;
}

int
param_is_nil(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    return ps->value[p->index].nil;
}

double
param_get_double(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if (param_is_nil(ps, name)) {
        message(0, "Accessing an undefined parameter `%s`.\n", p->name);
    }
    return ps->value[p->index].d;
}

char *
param_get_string(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if (param_is_nil(ps, name)) {
        message(0, "Accessing an undefined parameter `%s`.\n", p->name);
    }
    return ps->value[p->index].s;
}
void
param_get_string2(ParameterSet * ps, const char * name, char * dst, size_t len)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if (param_is_nil(ps, name)) {
        message(0, "Accessing an undefined parameter `%s`.\n", p->name);
    }
    if(strlen(ps->value[p->index].s) > len)
        endrun(1, "Parameter string %s too long for storage (%ld)\n", ps->value[p->index].s, len);
    strncpy(dst, ps->value[p->index].s, len);
    dst[len-1]='\0';
}

int
param_get_int(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if (param_is_nil(ps, name)) {
        message(0, "Accessing an undefined parameter `%s`.\n", p->name);
    }
    return ps->value[p->index].i;
}

int
param_get_enum(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if (param_is_nil(ps, name)) {
        message(0, "Accessing an undefined parameter `%s`.\n", p->name);
    }
    return ps->value[p->index].i;
}

char *
param_format_value(ParameterSet * ps, const char * name)
{
    ParameterSchema * p = param_get_schema(ps, name);
    if(ps->value[p->index].nil) {
        return fastpm_strdup("UNDEFINED");
    }
    switch(p->type) {
        case INT:
        {
            int i = ps->value[p->index].i;
            char buf[128];
            snprintf(buf, 128, "%d", i);
            return fastpm_strdup(buf);
        }
        break;
        case DOUBLE:
        {
            double d = ps->value[p->index].d;
            char buf[128];
            snprintf(buf, 128, "%g", d);
            return fastpm_strdup(buf);
        }
        break;
        case STRING:
        {
            return fastpm_strdup(ps->value[p->index].s);
        }
        break;
        case ENUM:
        {
            return format_enum(p->enumtable, ps->value[p->index].i);
        }
        break;
    }
    return fastpm_strdup("UNDEFINED TYPE");
}

void
param_set_from_string(ParameterSet * ps, const char * name, char * value, int lineno)
{
    ParameterSchema * p = param_get_schema(ps, name);
    ps->value[p->index].lineno = lineno;
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
            while(*value == ' ') value ++;
            ps->value[p->index].s = fastpm_strdup(value);
            char * a = ps->value[p->index].s;
            a += strlen(a) - 1;
            while(*a == ' ') {
                *a = 0;
                a--;
            }
            ps->value[p->index].nil = 0;
        }
        break;
        case ENUM:
        {
            char * v = fastpm_strdup(value);
            ps->value[p->index].i = parse_enum(p->enumtable, v);
            myfree(v);
            ps->value[p->index].nil = 0;
        }
        break;
    }
}

ParameterSet *
parameter_set_new()
{
    ParameterSet * ps = ta_malloc("paramset", ParameterSet, 1);
    ps->size = 0;
    return ps;
}

void
parameter_set_free(ParameterSet * ps) {
    /* Just reset the temp heap,
     * since that is where the paramset is allocated*/
    ta_reset();
    /*    int i;
     for(i = 0; i < ps->size; i ++) {
        if(ps->p[i].help) {
            myfree(ps->p[i].help);
        }
        if(ps->p[i].type == STRING) {
            if(ps->p[i].defvalue.s) {
                // FIXME: memory corruption
//                free(ps->p[i].defvalue.s);
            }
            if(ps->value[ps->p[i].index].s != ps->p[i].defvalue.s) {
                if(ps->value[ps->p[i].index].s) {
//                    free(ps->value[ps->p[i].index].s);
                }
            }
        }
    }
    myfree(ps);
    */
}

