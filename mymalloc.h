/*! \file mymalloc.h
 *  \brief Declares globally accessible structures and functions related to gadget's internal memory manager.
 */

#ifndef MALLOC_H
#define MALLOC_H
void *mymalloc_fullinfo(const char *varname, size_t n, const char *func, const char *file, int linenr);
void *mymalloc_movable_fullinfo(void *ptr, const char *varname, size_t n, const char *func, const char *file, int line);

void *myrealloc_fullinfo(void *p, size_t n, const char *func, const char *file, int line);
void *myrealloc_movable_fullinfo(void *p, size_t n, const char *func, const char *file, int line);

void myfree_fullinfo(void *p, const char *func, const char *file, int line);
void myfree_movable_fullinfo(void *p, const char *func, const char *file, int line);

void mymalloc_init(void);
void dump_memory_table(void);
void report_detailed_memory_usage_of_largest_task(const char *label, const char *func, const char *file, int line);

extern size_t AllocatedBytes;
extern size_t FreeBytes;

#define  mymalloc(x, y)            mymalloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define  mymalloc_movable(x, y, z) mymalloc_movable_fullinfo(x, y, z, __FUNCTION__, __FILE__, __LINE__)

#define  myrealloc(x, y)           myrealloc_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)
#define  myrealloc_movable(x, y)   myrealloc_movable_fullinfo(x, y, __FUNCTION__, __FILE__, __LINE__)

#define  myfree(x)                 (myfree_fullinfo(x, __FUNCTION__, __FILE__, __LINE__), x = NULL)
#define  myfree_movable(x)         (myfree_movable_fullinfo(x, __FUNCTION__, __FILE__, __LINE__), x = NULL)

#define  report_memory_usage(x) report_detailed_memory_usage_of_largest_task(x, __FUNCTION__, __FILE__, __LINE__)

#endif
