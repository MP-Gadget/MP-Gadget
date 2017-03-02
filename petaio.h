#ifndef PETAIO_H
#define PETAIO_H

#include "bigfile.h"

typedef void (*property_getter) (int i, void * result);
typedef void (*property_setter) (int i, void * target);
typedef int (*petaio_selection) (int i);

typedef struct IOTableEntry {
    char name[64];
    int ptype;
    char dtype[8];
    int items;
    property_getter getter;
    property_setter setter;
} IOTableEntry;

#define PTYPE_FOF_GROUP  1024

void petaio_init();
void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t npartLocal);
void petaio_build_buffer(BigArray * array, IOTableEntry * ent, int * selection, int NumSelection);
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent);
void petaio_destroy_buffer(BigArray * array);

void petaio_save_block(BigFile * bf, char * blockname, BigArray * array);
void petaio_read_block(BigFile * bf, char * blockname, BigArray * array);

void petaio_save_snapshot(const char * fmt, ...);
void petaio_save_restart();
void petaio_read_snapshot(int num);
void petaio_read_ic();
/* 
 * Declares a io block with name (literal, not a string) 
 * 
 * will use GT ## name and PT ## name for getter and putter.
 * these functions shall be declared in the module IO_REG is called.
 * 
 * SIMPLE_GETTER defines a simple getter reading property from global particle
 * arrays.
 * */
#define IO_REG(name, dtype, items, ptype) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name , (property_setter) ST ## name)
#define IO_REG_WRONLY(name, dtype, items, ptype) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name , NULL)
void io_register_io_block(char * name, 
        char * dtype, 
        int items, 
        int ptype, 
        property_getter getter,
        property_setter setter
        );


/* 
 * define a simple getter function
 *
 * field: for example, P[i].Pos[0].
 *     i can be used to refer to the index of the particle being read.
 *
 * type:  a C type descr, float / double / int, it descirbes the
 *     expected format of the output buffer; (compiler knows the format of field)
 *
 * items: number of items in one property. 1 for scalars.  (3 for pos, eg)
 *
 * */
#define SIMPLE_GETTER(name, field, type, items) \
static void name(int i, type * out) { \
    int k; \
    for(k = 0; k < items; k ++) { \
        out[k] = *(&(field) + k); \
    } \
}
#define SIMPLE_SETTER(name, field, type, items) \
static void name(int i, type * out) { \
    int k; \
    for(k = 0; k < items; k ++) { \
        *(&(field) + k) = out[k]; \
    } \
}
#define SIMPLE_PROPERTY(name, field, type, items) \
    SIMPLE_GETTER(GT ## name , field, type, items) \
    SIMPLE_SETTER(ST ## name , field, type, items) \
/* 
 * currently 4096 entries are supported 
 * */
extern struct IOTable {
    IOTableEntry ent[4096];
    int used;
} IOTable;

#endif
