
void init_endrun(int backtrace);

void endrun(int where, const char * fmt, ...) __attribute__ ((noreturn)) ;
void message(int where, const char * fmt, ...);
