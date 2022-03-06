#ifndef PETAIO_H
#define PETAIO_H

#include <mpi.h>
#include "bigfile.h"
#include "utils/paramset.h"
#include "partmanager.h"
#include "slotsmanager.h"

/* Store parameters for unit conversions
 * on write*/
struct conversions
{
    double atime;
    double hubble;
};

typedef void (*property_getter) (int i, void * result, void * baseptr, void * slotptr, const struct conversions * params);
typedef void (*property_setter) (int i, void * target, void * baseptr, void * slotptr, const struct conversions * params);
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
void register_io_blocks(struct IOTable * IOTable, int WriteGroupID);
/* Write (but don't read) some extra output blocks useful for debugging the particle structure*/
void register_debug_io_blocks(struct IOTable * IOTable);
/* Free the entries in the IOTable.*/
void destroy_io_blocks(struct IOTable * IOTable);

void set_petaio_params(ParameterSet *ps);
int GetUsePeculiarVelocity(void);
void petaio_init();
void petaio_alloc_buffer(BigArray * array, IOTableEntry * ent, int64_t npartLocal);
void petaio_build_buffer(BigArray * array, IOTableEntry * ent, const int * selection, const int NumSelection, struct particle_data * Parts, struct slots_manager_type * SlotsManager, struct conversions * conv);
void petaio_readout_buffer(BigArray * array, IOTableEntry * ent, struct conversions * conv);
void petaio_destroy_buffer(BigArray * array);

void petaio_save_block(BigFile * bf, const char * blockname, BigArray * array, int verbose);
int petaio_read_block(BigFile * bf, const char * blockname, BigArray * array, int required);

void petaio_save_snapshot(struct IOTable * IOTable, int verbose, const double atime, const char *fmt, ...);
void petaio_read_snapshot(int num, const double atime, MPI_Comm Comm);
void petaio_read_header(int num);

void
petaio_build_selection(int * selection,
    int * ptype_offset,
    int * ptype_count,
    const struct particle_data * Parts,
    const int NumPart,
    int (*select_func)(int i, const struct particle_data * Parts)
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
void io_register_io_block(const char * name,
        const char * dtype,
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
 * type:  a C type descr, float / double / int, it describes the
 *     expected format of the output buffer; (compiler knows the format of field)
 *
 * items: number of items in one property. 1 for scalars.  (3 for pos, eg)
 *
 * stype: type of the base pointer to use
 * */
#define SIMPLE_GETTER(name, field, type, items, stype) \
static void name(int i, type * out, void * baseptr, void * slotptr, const struct conversions * params) { \
    int k; \
    for(k = 0; k < items; k ++) { \
        out[k] = *(&(((stype *)baseptr)[i].field) + k); \
    } \
}
#define SIMPLE_SETTER(name, field, type, items, stype) \
static void name(int i, type * out, void * baseptr, void * slotptr, const struct conversions * params) { \
    int k; \
    for(k = 0; k < items; k ++) { \
        *(&(((stype *)baseptr)[i].field) + k) = out[k]; \
    } \
}

#endif
