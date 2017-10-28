#ifndef _MEMORY_H_
#define _MEMORY_H_

typedef struct Allocator Allocator;

#define ALLOC_ENOTALLOC -3
#define ALLOC_EMISMATCH -2
#define ALLOC_ENOMEMORY -1

#define ALLOC_DIR_TOP -1
#define ALLOC_DIR_BOT +1
#define ALLOC_DIR_BOTH 0

struct Allocator {
    char name[12];
    Allocator * parent;

    void * rawbase;
    void * base;
    size_t size;

    size_t bottom;
    size_t top;

    int refcount;
    int use_malloc; /* only do the book keeping. delegate to libc malloc/free */
};

typedef struct AllocatorIter AllocatorIter;
struct AllocatorIter {
    Allocator * alloc;
    size_t _bottom;
    size_t _top;
    int _ended;

    /* current block */
    size_t size;
    size_t request_size;
    char * name;
    int dir;
    char * annotation;
    void * ptr;
};

int
allocator_init(Allocator * alloc, char * name, size_t size, int zero, Allocator * parent);

int
allocator_malloc_init(Allocator * alloc,
        char * name, size_t size, int zero, Allocator * parent
        );

int
allocator_split(Allocator * alloc, Allocator * parent, char * name, size_t request_size, int zero);

int
allocator_destroy(Allocator * alloc);

void *
allocator_alloc(Allocator * alloc, char * name, size_t size, int dir, char * fmt, ...);

void *
allocator_realloc_int(Allocator * alloc, void * ptr, size_t size, char * fmt, ...);

#define allocator_alloc_bot(alloc, name, size) \
    allocator_alloc(alloc, name, size, ALLOC_DIR_BOT, "%s:%d", __FILE__, __LINE__)

#define allocator_alloc_top(alloc, name, size) \
    allocator_alloc(alloc, name, size, ALLOC_DIR_TOP, "%s:%d", __FILE__, __LINE__)

#define allocator_realloc(alloc, ptr, size) \
    allocator_realloc_int(alloc, ptr, size, "%s:%d", __FILE__, __LINE__)

/* free like API, will look up allocator pointer. */
void
allocator_free(void * ptr);

int
allocator_dealloc (Allocator * alloc, void * ptr);

size_t
allocator_get_free_size(Allocator * alloc);

int
allocator_iter_ended(AllocatorIter * iter);

int
allocator_iter_next(
        AllocatorIter * iter
    );

int
allocator_iter_start(
        AllocatorIter * iter,
        Allocator * alloc
    );

/* 0 for total */
size_t
allocator_get_used_size(Allocator * alloc, int dir);

void
allocator_print(Allocator * alloc);

int
allocator_reset(Allocator * alloc, int zero);

#endif
