#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdarg.h>

#include <sys/stat.h>
#include <unistd.h>

#include "string.h"
#include "mymalloc.h"

char *
fastpm_file_get_content(const char * filename)
{
    FILE * fp = fopen(filename, "r");
    if(!fp) return NULL;

    fseek(fp, 0, SEEK_END);
    size_t file_length = ftell(fp);

    char * buf = ta_malloc2(filename, char, file_length + 1);
    fseek(fp, 0, SEEK_SET);
    file_length = fread(buf, 1, file_length, fp);
    fclose(fp);
    buf[file_length] = 0;
    return buf;
}

char *
fastpm_strdup(const char * str)
{
    size_t N = strlen(str);
    char * d = ta_malloc("strdup", char, N + 1);
    strcpy(d, str);
    d[N] = '\0';
    return d;
}

char *
fastpm_strdup_printf(const char * fmt, ...)
{
    va_list va;
    va_start(va, fmt);
    char * buf = fastpm_strdup_vprintf(fmt, va);
    va_end(va);
    return buf;
}

char *
fastpm_strdup_vprintf(const char * fmt, va_list va)
{
    va_list va2;
    va_copy(va2, va);
    /* This relies on a good LIBC vsprintf that returns the number of char */
    char buf0[128];
    size_t N = vsnprintf(buf0, 1, fmt, va);

    char * buf = ta_malloc("strdup", char, N + 100);
    vsnprintf(buf, N + 1, fmt, va2);
    buf[N + 1] = 0;
    va_end(va2);
    return buf;
}

static void
_mkdir(const char *dir);

void
fastpm_path_ensure_dirname(const char * path)
{
    int i = strlen(path);
    char * dup = ta_malloc("dirname", char, strlen(path) + 1);
    strcpy(dup, path);
    dup[strlen(path)]='\0';
    char * p;
    for(p = i + dup; p >= dup && *p != '/'; p --) {
        continue;
    }
    /* plain file name in current directory */
    if(p < dup) return;

    /* p == '/', so set it to NULL, dup is the dirname */
    *p = 0;
    _mkdir(dup);
    myfree(dup);
}

static void
_mkdir(const char *dir)
{
    char * tmp= ta_malloc("dirname", char, strlen(dir) + 1);
    strcpy(tmp, dir);
    tmp[strlen(dir)]='\0';
    char *p = NULL;
    size_t len;

    len = strlen(tmp);
    if(tmp[len - 1] == '/')
            tmp[len - 1] = 0;
    for(p = tmp + 1; *p; p++)
            if(*p == '/') {
                    *p = 0;
                    mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO);
                    *p = '/';
            }
    mkdir(tmp, S_IRWXU | S_IRWXG | S_IRWXO);
    myfree(tmp);
}


