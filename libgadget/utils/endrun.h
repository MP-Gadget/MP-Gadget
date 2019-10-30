#ifndef _ENDRUN_H
#define _ENDRUN_H

void init_endrun(int backtrace);

void endrun(int where, const char * fmt, ...) __attribute__ ((noreturn)) ;
void message(int where, const char * fmt, ...);

#endif
