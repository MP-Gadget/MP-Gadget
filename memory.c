#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "memory.h"
#include "endrun.h"

#define MAGIC "DEADBEEF"
#define ALIGNMENT 4096

struct BlockHeader {
    char magic[8];
    Allocator * alloc;
    void * ptr;
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
    if (base == NULL) return ALLOC_ENOMEMORY;

    strncpy(alloc->name, name, 11);
    alloc->base = base;
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
allocator_alloc(Allocator * alloc, char * name, size_t size, int dir, char * fmt, ...)
{
    size_t request_size = size;
    size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    size += ALIGNMENT; /* for the header */
    void * ptr;
    if(dir == ALLOC_DIR_BOT) {
        if(alloc->bottom + size > alloc->top) {
            allocator_print(alloc);
            abort();
        }
        ptr = alloc->base + alloc->bottom;
        alloc->bottom += size;
        alloc->refcount += 1;
    } else if (dir == ALLOC_DIR_TOP) {
        if(alloc->top < alloc->bottom + size) {
            allocator_print(alloc);
            abort();
        }
        ptr = alloc->base + alloc->top - size;
        alloc->refcount += 1;
        alloc->top -= size;
    } else {
        /* wrong dir cannot allocate */
        return NULL;
    }

    struct BlockHeader * header = ptr;
    memcpy(header->magic, MAGIC, 8);
    header->size = size;
    header->request_size = request_size;
    header->dir = dir;
    header->alloc = alloc;
    strncpy(header->name, name, 127);
    va_list va;
    va_start(va, fmt);
    vsprintf(header->annotation, fmt, va);
    va_end(va);
    char * cptr = ptr;
    cptr += ALIGNMENT;
    header->ptr = cptr;
    return (void*) (cptr);
}

int
allocator_destroy(Allocator * alloc)
{
    if(alloc->refcount != 1) {
        allocator_print(alloc);
        endrun(1, "leaked\n");
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

static int
is_header(struct BlockHeader * header)
{
    return 0 == memcmp(header->magic, MAGIC, 8);
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
    if(iter->_top != alloc->size) {
        header = iter->_top + alloc->base;
        iter->_top += header->size;
    } else {
        iter->_ended = 1;
        return 0;
    }
    if (! is_header(header)) {
        /* several corruption that shall not happen */
        abort();
    }
    iter->ptr =  header->name;
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

size_t
allocator_get_used_size(Allocator * alloc, int dir)
{
    if (dir == ALLOC_DIR_TOP) {
        return (alloc->size - alloc->top);
    }
    if (dir == ALLOC_DIR_BOT) {
        return (alloc->bottom - 0);
    }
    if (dir == ALLOC_DIR_BOTH) {
        return (alloc->size - alloc->top + alloc->bottom - 0);
    }
    /* unknown */
    return 0;
}

void
allocator_print(Allocator * alloc)
{
    message(1, "--------------- Allocator: %s | Total: %010td bytes -----------------\n",
                alloc->name, alloc->size);
    message(1, "Free: %010td | Used: %010td Top: %010td Bottom: %010td \n",
            allocator_get_free_size(alloc),
            allocator_get_used_size(alloc, ALLOC_DIR_BOTH),
            allocator_get_used_size(alloc, ALLOC_DIR_TOP),
            allocator_get_used_size(alloc, ALLOC_DIR_BOT)
            );
    AllocatorIter iter[1];
    for(allocator_iter_start(iter, alloc);
        !allocator_iter_ended(iter);
        allocator_iter_next(iter))
    {
        message(1, "%08p | %-20s | %c %010td %010td | %s\n", 
                 iter->ptr,
                 iter->name,
                 "T?B"[iter->dir + 1],
                 iter->request_size, iter->size, iter->annotation);
    }
}

void
allocator_free (void * ptr)
{
    char * cptr = ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);

    if (!is_header(header)) {
        allocator_print(header->alloc);
        endrun(1, "Not an allocated address: Header = %08p ptr = %08p\n", header, cptr);
    }

    int rt = allocator_dealloc(header->alloc, ptr);
    if (rt != 0) {
        allocator_print(header->alloc);
        endrun(1, "Mismatched Free: %s : %s\n", header->name, header->annotation);
    }
}

int
allocator_dealloc (Allocator * alloc, void * ptr)
{
    char * cptr = ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);
    if (!is_header(header)) {
        return ALLOC_ENOTALLOC;
    }

    ptr = header;
    if(header->dir == ALLOC_DIR_BOT) {
        if(ptr != alloc->bottom - header->size + alloc->base) {
            return ALLOC_EMISMATCH;
        }
        alloc->bottom -= header->size;
    } else if(header->dir == ALLOC_DIR_TOP) {
        if(ptr != alloc->top + alloc->base) {
            return ALLOC_EMISMATCH;
        }
        alloc->top += header->size;
    } else {
        return ALLOC_ENOTALLOC;
    }

    /* remove the link to the memory. */
    header->ptr = NULL;
    header->alloc = NULL;
    alloc->refcount --;
    return 0;
}
