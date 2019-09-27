#ifndef PETAIO_H
#define PETAIO_H

#include <mpi.h>
#include "bigfile.h"

typedef void (*property_getter) (int i, void * result);
typedef void (*property_setter) (int i, void * target);
typedef int (*petaio_selection) (int i);

typedef struct IOTableEntry {
    int zorder;
    char name[64];
    int ptype;
    char dtype[8];
    int items;
    int required;
    property_getter getter;
    property_setter setter;
} IOTableEntry;

struct IOTable {
    IOTableEntry * ent;
    int used;
    int allocated;
};

#define PTYPE_FOF_GROUP  1024

/* Populate an IOTable with the default set of blocks to read or write.*/
void register_io_blocks(struct IOTable * IOTable);
/* Write (but don't read) some extra output blocks useful for debugging the particle structure*/
void register_debug_io_blocks(struct IOTable * IOTable);
/* Free the entries in the IOTable.*/
void destroy_io_blocks(struct IOTable * IOTable);

void petaio_init();
void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t npartLocal);
void petaio_build_buffer(BigArray * array, IOTableEntry * ent, const int * selection, const int size);
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent);
void petaio_destroy_buffer(BigArray * array);

void petaio_save_block(BigFile * bf, char * blockname, BigArray * array);
int petaio_read_block(BigFile * bf, char * blockname, BigArray * array, int required);

void petaio_save_snapshot(struct IOTable * IOTable, const char *fmt, ...);
void petaio_read_snapshot(int num, MPI_Comm Comm);
void petaio_read_header(int num);

void
petaio_build_selection(int * selection,
    int * ptype_offset,
    int * ptype_count,
    const int NumPart,
    int (*select_func)(int i)
    );
/*
 * Declares a io block with name (literal, not a string)
 *
 * will use GT ## name and PT ## name for getter and putter.
 * these functions shall be declared in the module IO_REG is called.
 *
 * SIMPLE_GETTER defines a simple getter reading property from global particle
 * arrays.
 *
 * IO_REG_TYPE declares an io block which has a type-specific property setter.
 * IO_REG_WRONLY declares an io block which is written, but is not read on snapshot load.
 * */
#define IO_REG(name, dtype, items, ptype, IOTable) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name , (property_setter) ST ## name, 1, IOTable)
#define IO_REG_TYPE(name, dtype, items, ptype, IOTable) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## ptype ## name , (property_setter) ST ## ptype ## name, 0, IOTable)
#define IO_REG_WRONLY(name, dtype, items, ptype, IOTable) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name , NULL, 1, IOTable)
#define IO_REG_NONFATAL(name, dtype, items, ptype, IOTable) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name , (property_setter) ST ## name, 0, IOTable)
void io_register_io_block(char * name,
        char * dtype,
        int items,
        int ptype,
        property_getter getter,
        property_setter setter,
        int required,
        struct IOTable * IOTable
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
/*A property with getters and setters that are type specific*/
#define SIMPLE_PROPERTY_TYPE(name, ptype, field, type, items) \
    SIMPLE_GETTER(GT ## ptype ## name , field, type, items) \
    SIMPLE_SETTER(ST ## ptype ## name , field, type, items) \

#endif
