#ifndef _FASTPM_STRING_H_
#define _FASTPM_STRING_H_

#include <stdarg.h>

char *
fastpm_file_get_content(const char * filename);

char *
fastpm_strdup(const char * str);

char *
fastpm_strdup_printf(const char * fmt, ...);

char *
fastpm_strdup_vprintf(const char * fmt, va_list va);

void
fastpm_path_ensure_dirname(const char * path);

#endif
