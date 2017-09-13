#ifndef _MYMALLOC_H_
#define _MYMALLOC_H_

#include "memory.h"

extern Allocator A_MAIN[1];
extern Allocator A_TEMP[1];

void mymalloc_init(double MemoryMB);
void report_detailed_memory_usage(const char *label, const char * fmt, ...);

#define  mymalloc(x, y)            allocator_alloc_bot(A_MAIN, x, y)
#define  mymalloc2(x, y)            allocator_alloc_top(A_MAIN, x, y)
#define  myfree(x)                 allocator_free(x)
#define  report_memory_usage(x)    report_detailed_memory_usage(x, "%s:%d", __FILE__, __LINE__)
#define  FreeBytes                 allocator_get_free_size(A_MAIN)
#define  AllocatedBytes            allocator_get_used_size(A_MAIN, ALLOC_DIR_BOTH)
#endif
