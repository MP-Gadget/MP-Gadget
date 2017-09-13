#include <stdlib.h>
#include <string.h>
#include "memory.h"
#include "endrun.h"

#define ALIGNMENT 4096

struct BlockHeader {
    size_t size;
    size_t request_size;
    char name[127];
    int dir;
    char annotation[];
} ;

int
allocator_init(Allocator * alloc, char * name, size_t size, int zero)
{
    void * base = malloc(size);
    if (base == NULL) return -1;

    strncpy(alloc->name, name, 11);
    alloc->refcount = 1;
    alloc->size = size;
    alloc->bottom = 0;
    alloc->top = size;

    if(zero) {
        memset(alloc->base, 0, size);
    }
    return 0;

}

void *
allocator_alloc(Allocator * alloc, char * name, size_t size, int zero, int dir, char * annotation)
{
    size_t request_size = size;
    size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    size += ALIGNMENT; /* for the header */
    void * ptr;
    if(dir > 0) {
        if(alloc->bottom + size > alloc->top) {
            allocator_print(alloc);
            abort();
        }
        ptr = alloc->base + alloc->bottom;
        alloc->bottom += size;
        alloc->refcount += 1;
    } else {
        if(alloc->top < alloc->bottom + size) {
            allocator_print(alloc);
            abort();
        }
        ptr = alloc->base + alloc->top - size;
        alloc->refcount += 1;
        alloc->top -= size;
    }

    struct BlockHeader * header = ptr;
    header->size = size;
    header->request_size = request_size;
    header->dir = dir;
    strncpy(header->name, name, 127);
    strncpy(header->annotation, annotation, 2048);
    char * cptr = ptr;
    return (void*) (cptr + ALIGNMENT);
}

int
allocator_destroy(Allocator * alloc)
{
    if(alloc->refcount != 0) {
        abort();
    }
    free(alloc->base);
}

int
allocator_iter_start(
        AllocatorIter * iter,
        Allocator * alloc
    )
{
    iter->alloc = alloc;
    iter->_bottom = 0;
    iter->_top = alloc->top;
    iter->_ended = 0;
    return allocator_iter_next(iter);
}

int
allocator_iter_next(
        AllocatorIter * iter
    )
{
    struct BlockHeader * header;
    Allocator * alloc = iter->alloc;
    if(alloc->bottom != iter->_bottom) {
        header = iter->_bottom + alloc->base;
        iter->_bottom += header->size;
    } else
    if(alloc->top != alloc->size) {
        header = iter->_top + alloc->base;
        iter->_top += header->size;
    } else {
        iter->_ended = 1;
        return 0;
    }
    iter->name = header->name;
    iter->annotation = header->annotation;
    iter->size = header->size;
    iter->request_size = header->request_size;
    iter->dir = header->dir;
    return 1;
}

int
allocator_iter_ended(AllocatorIter * iter)
{
    return iter->_ended;
}

size_t
allocator_get_free_size(Allocator * alloc)
{
    return (alloc->top - alloc->bottom);
}

void
allocator_print(Allocator * alloc)
{
    message(1, "--------------- %s -----------------\n", alloc->name);
    message(1, "%td bytes free\n", allocator_get_free_size(alloc));
    AllocatorIter iter[1];
    for(allocator_iter_start(iter, alloc);
        !allocator_iter_ended(iter);
        allocator_iter_next(iter))
    {
        message(1, "% 20s %09td / %09td %s\n", iter->name, iter->request_size, iter->size, iter->annotation);
    }
}

void
allocator_free (Allocator * alloc, void * ptr)
{
    char * cptr = ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);
    ptr = header;
    if(header->dir > 0) {
        alloc->bottom -= header->size;
        if(ptr != alloc->bottom + alloc->base) {
            allocator_print(alloc);
            abort();
        }
    } else {
        alloc->top += header->size;
        if(ptr != alloc->top + alloc->base - header->size) {
            allocator_print(alloc);
            abort();
        }
    }
    alloc->refcount --;
}
