#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "memory.h"
#include "endrun.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#define MAGIC "DEADBEEF"
#define ALIGNMENT 4096

struct BlockHeader {
    char magic[8];
    Allocator * alloc;
    char * ptr;
    char * self; /* points to the starting of the header in the allocator; useful in use_malloc mode */
    size_t size;
    size_t request_size;
    char name[127];
    int dir;
    char annotation[];
} ;

int
allocator_init(Allocator * alloc, const char * name, const size_t request_size, const int zero, Allocator * parent)
{
    size_t size = (request_size / ALIGNMENT + 1) * ALIGNMENT;

    char * rawbase;
    if (parent) {
        rawbase = (char *) allocator_alloc(parent, name, size + ALIGNMENT, ALLOC_DIR_BOT, "Child");
        if(rawbase == NULL)
            return ALLOC_ENOMEMORY;
    }
    else
        if(posix_memalign((void **) &rawbase, ALIGNMENT, size + ALIGNMENT))
            return ALLOC_ENOMEMORY;

    alloc->parent = parent;
    alloc->rawbase = rawbase;
    alloc->base = rawbase + ALIGNMENT - ((size_t) rawbase % ALIGNMENT);
    alloc->size = size;
    alloc->use_malloc = 0;
    strncpy(alloc->name, name, 11);
    alloc->refcount = 1;
    alloc->top = alloc->size;
    alloc->bottom = 0;

    allocator_reset(alloc, zero);

    return 0;
}

int
allocator_malloc_init(Allocator * alloc, const char * name, const size_t request_size, const int zero, Allocator * parent)
{
    /* max support 4096 blocks; ignore request_size */
    size_t size = ALIGNMENT * 4096;

    char * rawbase;
    if (parent) {
        rawbase = (char *) allocator_alloc(parent, name, size + ALIGNMENT, ALLOC_DIR_BOT, "Child");
        if (rawbase == NULL) return ALLOC_ENOMEMORY;
    }
    else
        if(posix_memalign((void **) &rawbase, ALIGNMENT, size + ALIGNMENT))
            return ALLOC_ENOMEMORY;


    alloc->parent = parent;
    alloc->use_malloc = 1;
    alloc->rawbase = rawbase;
    alloc->base = rawbase;
    alloc->size = size;
    strncpy(alloc->name, name, 11);
    alloc->refcount = 1;
    alloc->top = alloc->size;
    alloc->bottom = 0;

    allocator_reset(alloc, zero);

    return 0;
}

int
allocator_reset(Allocator * alloc, int zero)
{
    /* Free the memory when using malloc*/
    if(alloc->use_malloc) {
        AllocatorIter iter[1];
        for(allocator_iter_start(iter, alloc); !allocator_iter_ended(iter); allocator_iter_next(iter))
        {
            free(iter->ptr - ALIGNMENT);
        }
    }
    alloc->refcount = 1;
    alloc->top = alloc->size;
    alloc->bottom = 0;

    if(zero) {
        memset(alloc->base, 0, alloc->size);
    }
    return 0;
}

static void *
allocator_alloc_va(Allocator * alloc, const char * name, const size_t request_size, const int dir, const char * fmt, va_list va)
{
    size_t size = request_size;

    if(alloc->use_malloc) {
        size = 0; /* because we'll get it from malloc */
    } else {
        size = ((size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
    }
    size += ALIGNMENT; /* for the header */

    char * ptr;
    if(dir == ALLOC_DIR_BOT) {
        if(alloc->bottom + size > alloc->top) {
            allocator_print(alloc);
            endrun(1, "Not enough memory for %s %lu bytes\n", name, size);
        }
        ptr = alloc->base + alloc->bottom;
        alloc->bottom += size;
        alloc->refcount += 1;
    } else if (dir == ALLOC_DIR_TOP) {
        if(alloc->top < alloc->bottom + size) {
            allocator_print(alloc);
            endrun(1, "Not enough memory for %s %lu bytes\n", name, size);
        }
        ptr = alloc->base + alloc->top - size;
        alloc->refcount += 1;
        alloc->top -= size;
    } else {
        /* wrong dir cannot allocate */
        return NULL;
    }

    struct BlockHeader * header = (struct BlockHeader *) ptr;
    memcpy(header->magic, MAGIC, 8);
    header->self = ptr;
    header->size = size;
    header->request_size = request_size;
    header->dir = dir;
    header->alloc = alloc;
    strncpy(header->name, name, 126);
    header->name[126] = '\0';

    vsprintf(header->annotation, fmt, va);

    char * cptr;
    if(alloc->use_malloc) {
        /* prepend a copy of the header to the malloc block; allocator_free will use it*/
    #ifdef USE_CUDA
        if (cudaMallocManaged((void **) &cptr, request_size + ALIGNMENT, cudaMemAttachGlobal) != cudaSuccess)
    #else
        if(posix_memalign((void **) &cptr, ALIGNMENT, request_size + ALIGNMENT))
    #endif
            endrun(1, "Failed malloc: %lu bytes for %s\n", request_size, header->name);
        header->ptr = cptr + ALIGNMENT;
        memcpy(cptr, header, ALIGNMENT);
        cptr = header->ptr;
    } else {
        cptr = ptr + ALIGNMENT;
        header->ptr = cptr;
    }
    return cptr;
}
void *
allocator_alloc(Allocator * alloc, const char * name, const size_t request_size, const int dir, const char * fmt, ...)
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
        header = (struct BlockHeader *) (iter->_bottom + alloc->base);
        iter->_bottom += header->size;
    } else
    if(iter->_top != alloc->size) {
        header = (struct BlockHeader *) (iter->_top + alloc->base);
        iter->_top += header->size;
    } else {
        iter->_ended = 1;
        return 0;
    }
    if (! is_header(header)) {
        /* several corruption that shall not happen */
        endrun(5, "Ptr %p is not a magic header\n", header);
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
    /*For malloc, return a fixed 2GB */
    if(alloc->use_malloc) {
        return 2L*1024L*1024L*1024L;
    }
    return (alloc->top - alloc->bottom);
}

size_t
allocator_get_used_size(Allocator * alloc, int dir)
{
    /* For malloc sum up the requested memory.
     * I considered mallinfo, but there may be multiple memory arenas. */
    if(alloc->use_malloc) {
        size_t total = 0;
        AllocatorIter iter[1];
        for(allocator_iter_start(iter, alloc); !allocator_iter_ended(iter);
            allocator_iter_next(iter))
        {
            total += iter->request_size;
        }
        return total;
    }
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
    message(1, " Total: %010td kbytes\n", alloc->size/1024);
    message(1, " Free: %010td Used: %010td Top: %010td Bottom: %010td \n",
            allocator_get_free_size(alloc)/1024,
            allocator_get_used_size(alloc, ALLOC_DIR_BOTH)/1024,
            allocator_get_used_size(alloc, ALLOC_DIR_TOP)/1024,
            allocator_get_used_size(alloc, ALLOC_DIR_BOT)/1024
            );
    AllocatorIter iter[1];
    message(1, " %-20s | %c | %-12s %-12s | %s\n", "Name", 'd', "Requested", "Allocated", "Annotation");
    message(1, "-------------------------------------------------------\n");
    for(allocator_iter_start(iter, alloc);
        !allocator_iter_ended(iter);
        allocator_iter_next(iter))
    {
        message(1, " %-20s | %c | %012td %012td | %s\n",
                 iter->name,
                 "T?B"[iter->dir + 1],
                 iter->request_size/1024, iter->size/1024, iter->annotation);
    }
}

void *
allocator_realloc_int(Allocator * alloc, void * ptr, const size_t new_size, const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);

    char * cptr = (char *) ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);
    struct BlockHeader tmp = * header;

    if (!is_header(header)) {
        allocator_print(header->alloc);
        endrun(1, "Not an allocated address: Header = %8p ptr = %8p\n", header, cptr);
    }

    if(alloc->use_malloc) {
        struct BlockHeader * header2 = (struct BlockHeader *) realloc(header, new_size + ALIGNMENT);
        header2->ptr = (char*) header2 + ALIGNMENT;
        header2->request_size = new_size;
        /* update record */
        vsprintf(header2->annotation, fmt, va);
        va_end(va);
        memcpy(header2->self, header2, sizeof(header2[0]));
        return header2->ptr;
    }

    if(0 != allocator_dealloc(alloc, ptr)) {
        allocator_print(header->alloc);
        endrun(1, "Mismatched Free: %s : %s\n", header->name, header->annotation);
    }

    /*If we are shrinking memory, move the existing data block up and then write a new header.*/
    if(tmp.dir == ALLOC_DIR_TOP && new_size < tmp.request_size) {
        /*Offset for new memory, after header*/
        size_t size = ((new_size + ALIGNMENT - 1) / ALIGNMENT) * ALIGNMENT;
        memmove(alloc->base + alloc->top - size, tmp.ptr, new_size);
    }
    void * newptr = allocator_alloc_va(alloc, tmp.name, new_size, tmp.dir, fmt, va);
    /*If we are extending memory, move the existing data block down after writing a new header below it*/
    if(tmp.dir == ALLOC_DIR_TOP && new_size > tmp.request_size) {
        memmove(newptr, tmp.ptr, tmp.size);
    }
    va_end(va);
    return newptr;
}

void
allocator_free (void * ptr)
{
    char * cptr = (char *) ptr;
    struct BlockHeader * header = (struct BlockHeader*) (cptr - ALIGNMENT);

    if (!is_header(header)) {
        allocator_print(header->alloc);
        endrun(1, "Not an allocated address: Header = %8p ptr = %8p\n", header, cptr);
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
    char * cptr = (char *) ptr;
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
    header = (struct BlockHeader *) ptr; /* modify the true header in the allocator */
    header->ptr = NULL;
    header->self = NULL;
    header->alloc = NULL;
    alloc->refcount --;

    return 0;
}
