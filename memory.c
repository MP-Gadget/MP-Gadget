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
    void * self; /* points to the starting of the header in the allocator; useful in use_malloc mode */
    size_t size;
    size_t request_size;
    char name[127];
    int dir;
    char annotation[];
} ;

int
allocator_init(Allocator * alloc, char * name, size_t request_size, int zero, Allocator * parent)
{
    size_t size = (request_size / ALIGNMENT + 1) * ALIGNMENT;

    void * rawbase;
    if (parent)
        rawbase = allocator_alloc(parent, name, size + ALIGNMENT, ALLOC_DIR_BOT, "Child");
    else
        rawbase = malloc(size + ALIGNMENT);

    if (rawbase == NULL) return ALLOC_ENOMEMORY;

    alloc->parent = parent;
    alloc->rawbase = rawbase;
    alloc->base = ((char*) rawbase) + ALIGNMENT - ((size_t) rawbase % ALIGNMENT);
    alloc->size = size;
    alloc->use_malloc = 0;
    strncpy(alloc->name, name, 11);

    allocator_reset(alloc, zero);

    return 0;
}

int
allocator_malloc_init(Allocator * alloc, char * name, size_t request_size, int zero, Allocator * parent)
{
    /* max support 4096 blocks; ignore request_size */
    size_t size = ALIGNMENT * 4096; 

    void * rawbase;
    if (parent)
        rawbase = allocator_alloc(parent, name, size + ALIGNMENT, ALLOC_DIR_BOT, "Child");
    else
        rawbase = malloc(size + ALIGNMENT);

    if (rawbase == NULL) return ALLOC_ENOMEMORY;

    alloc->parent = parent;
    alloc->use_malloc = 1;
    alloc->rawbase = rawbase;
    alloc->base = rawbase;
    alloc->size = size;
    strncpy(alloc->name, name, 11);

    allocator_reset(alloc, zero);

    return 0;
}

int
allocator_reset(Allocator * alloc, int zero)
{
    alloc->refcount = 1;
    alloc->top = alloc->size;
    alloc->bottom = 0;

    if(zero) {
        memset(alloc->base, 0, alloc->size);
    }
    return 0;
}

static void *
allocator_alloc_va(Allocator * alloc, char * name, size_t request_size, int dir, char * fmt, va_list va)
{ 
    size_t size = request_size;

    if(alloc->use_malloc) {
        size = 0; /* because we'll get it from malloc */
    } else {
        size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    size += ALIGNMENT; /* for the header */

    void * ptr;
    if(dir == ALLOC_DIR_BOT) {
        if(alloc->bottom + size > alloc->top) {
            allocator_print(alloc);
            endrun(1, "Not enough memory for %s %td bytes\n", name, size);
        }
        ptr = alloc->base + alloc->bottom;
        alloc->bottom += size;
        alloc->refcount += 1;
    } else if (dir == ALLOC_DIR_TOP) {
        if(alloc->top < alloc->bottom + size) {
            allocator_print(alloc);
            endrun(1, "Not enough memory for %s %td bytes\n", name, size);
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
    header->self = ptr;
    header->size = size;
    header->request_size = request_size;
    header->dir = dir;
    header->alloc = alloc;
    strncpy(header->name, name, 127);

    vsprintf(header->annotation, fmt, va);

    char * cptr;
    if(alloc->use_malloc) {
        /* prepend a copy of the header to the malloc block; allocator_free will use it*/
        cptr = malloc(request_size + ALIGNMENT);
        header->ptr = cptr + ALIGNMENT;
        memcpy(cptr, header, ALIGNMENT);
        cptr += ALIGNMENT;
    } else {
        cptr = ptr + ALIGNMENT;
        header->ptr = cptr;
    }
    return (void*) (cptr);
}
void *
allocator_alloc(Allocator * alloc, char * name, size_t request_size, int dir, char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    void * rt = allocator_alloc_va(alloc, name, request_size, dir, fmt, va);
    va_end(va);
    return rt;
}

int
allocator_destroy(Allocator * alloc)
{
    if(alloc->refcount != 1) {
        allocator_print(alloc);
        endrun(1, "leaked\n");
    }
    if(alloc->parent)
        allocator_dealloc(alloc->parent, alloc->rawbase);
    else
        free(alloc->rawbase);
    return 0;
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
    iter->ptr =  header->ptr;
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
    message(1, "--------------- Allocator: %-17s %12s-----------------\n",
                alloc->name,
                alloc->use_malloc?"(libc managed)":"(self managed)"
                );
    message(1, " Total: %010td bytes\n", alloc->size);
    message(1, " Free: %010td Used: %010td Top: %010td Bottom: %010td \n",
            allocator_get_free_size(alloc),
            allocator_get_used_size(alloc, ALLOC_DIR_BOTH),
            allocator_get_used_size(alloc, ALLOC_DIR_TOP),
            allocator_get_used_size(alloc, ALLOC_DIR_BOT)
            );
    AllocatorIter iter[1];
    message(1, " %-20s | %c | %-10s %-10s | %s\n", "Name", 'd', "Requested", "Allocated", "Annotation");
    message(1, "-------------------------------------------------------\n");
    for(allocator_iter_start(iter, alloc);
        !allocator_iter_ended(iter);
        allocator_iter_next(iter))
    {
        message(1, " %-20s | %c | %010td %010td | %s\n", 
                 iter->name,
                 "T?B"[iter->dir + 1],
                 iter->request_size, iter->size, iter->annotation);
    }
}

void *
allocator_realloc_int(Allocator * alloc, void * ptr, size_t new_size, char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char * cptr = ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);
    struct BlockHeader tmp = * header;

    if (!is_header(header)) {
        allocator_print(header->alloc);
        endrun(1, "Not an allocated address: Header = %08p ptr = %08p\n", header, cptr);
    }

    if(alloc->use_malloc) {
        struct BlockHeader * header2 = realloc(header, new_size + ALIGNMENT);
        header2->ptr = (char*) header2 + ALIGNMENT;
        header2->request_size = new_size;
        /* update record */
        vsprintf(header2->annotation, fmt, va);
        va_end(va);
        memcpy(header2->self, header2, sizeof(header2[0]));
        return header2->ptr;
    }

    allocator_dealloc(alloc, ptr);
    void * newptr = allocator_alloc_va(alloc, tmp.name, new_size, tmp.dir, fmt, va);
    if(tmp.dir == ALLOC_DIR_TOP) {
        memmove(newptr, tmp.ptr, tmp.size);
    }
    va_end(va);
    return newptr;
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

    /* ->self is always the header in the allocator; header maybe a duplicate in use_malloc */
    ptr = header->self;
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

    if(alloc->use_malloc) {
        free(header);
    }

    /* remove the link to the memory. */
    header = ptr; /* modify the true header in the allocator */
    header->ptr = NULL;
    header->self = NULL;
    header->alloc = NULL;
    alloc->refcount --;

    return 0;
}
