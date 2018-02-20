#ifndef _MYMALLOC_H_
#define _MYMALLOC_H_

#include "memory.h"

extern Allocator A_MAIN[1];
extern Allocator A_TEMP[1];

void mymalloc_init(double MemoryMB);
void report_detailed_memory_usage(const char *label, const char * fmt, ...);

#define  mymalloc(name, size)            allocator_alloc_bot(A_MAIN, name, size)
#define  mymalloc2(name, size)           allocator_alloc_top(A_MAIN, name, size)

#define  myrealloc(ptr, size)     allocator_realloc(A_MAIN, ptr, size)
#define  myfree(x)                 allocator_free(x)

#define  ma_malloc(name, type, nele)            (type*) allocator_alloc_bot(A_MAIN, name, sizeof(type) * nele)
#define  ma_malloc2(name, type, nele)           (type*) allocator_alloc_top(A_MAIN, name, sizeof(type) * nele)
#define  ma_free(p) allocator_free(p)

#define  ta_malloc(name, type, nele)            (type*) allocator_alloc_bot(A_TEMP, name, sizeof(type) * nele)
#define  ta_malloc2(name, type, nele)           (type*) allocator_alloc_top(A_TEMP, name, sizeof(type) * nele)
#define  ta_reset()     allocator_reset(A_TEMP, 0)
#define  ta_free(p) allocator_free(p)

#define  report_memory_usage(x)    report_detailed_memory_usage(x, "%s:%d", __FILE__, __LINE__)
#define  FreeBytes                 allocator_get_free_size(A_MAIN)
#define  AllocatedBytes            allocator_get_used_size(A_MAIN, ALLOC_DIR_BOTH)

#endif
