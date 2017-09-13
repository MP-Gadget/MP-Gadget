#ifndef _MEMORY_H_
#define _MEMORY_H_

typedef struct Allocator Allocator;

struct Allocator {
    char name[12];
    Allocator * parent;

    void * base;
    size_t size;

    size_t bottom;
    size_t top;

    int refcount;
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
};

int
allocator_init(Allocator * alloc, char * name, size_t size, int zero);

int
allocator_destroy(Allocator * alloc);

void *
allocator_alloc(Allocator * alloc, char * name, size_t size, int zero, int dir, char * annotation);

void
allocator_free (Allocator * alloc, void * ptr);

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

void
allocator_print(Allocator * alloc);

#endif
