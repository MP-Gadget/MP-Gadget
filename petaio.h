typedef struct IOTableEntry IOTableEntry;
typedef void (*property_getter) (int i, void * result);
typedef struct IOTableEntry {
    char name[64];
    int ptype;
    char dtype[8];
    int items;
    property_getter getter;
} IOTableEntry;

#define PTYPE_FOF_GROUP  1024

void io_register_io_block(char * name, 
        char * dtype, 
        int items, 
        int ptype, 
        property_getter getter);

/* 
 * Decalres a io block with name (literal, not a string) 
 * 
 * will use GT ## name and PT ## name for getter and putter.
 * these functions shall be declared in the module IO_REG is called.
 * 
 * SIMPLE_GETTER defines a simple getter reading property from global particle
 * arrays.
 * */
#define IO_REG(name, dtype, items, ptype) \
    io_register_io_block(# name, dtype, items, ptype, (property_getter) GT ## name)

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

/* 
 * currently 4096 entries are supported 
 * */
extern struct IOTable {
    IOTableEntry ent[4096];
    int used;
} IOTable;
