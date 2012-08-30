CC     = mpicc

myINCLUDE = /afs/rzg/home/v/vrs/Libs/opteron64/mvapich2-0.9.3/include
myLIB     = /afs/rzg/home/v/vrs/Libs/opteron64/mvapich2-0.9.3/lib

FPIC      = "-fpic"
# ------------------------------------   default names for library
LIBIMF     = libIMF.so
LIBIMFS    = libIMF.a

# ------------------------------------   comment to avoid optimization; -ggdb -O0 will be used
#OPTIMIZE     = -O3
# ------------------------------------ define dependence files for each target
OBJS   = libimf_vars.o libimf.o
INCL   = libimf_vars.h Makefile
# ------------------------------------ define C FLAGS for compilation
CFLAGS = $(OPTIMIZE) $(FPIC) -m64 -ggdb -Wall -W -I$(myINCLUDE)
# ------------------------------------ useful stuffs
LNKCMD = ld -shared $(FPIC)
LNKCMDS= ar  rcs
LIBS   = -lm -L$(myLIB)
# ------------------------------------ define rules

libsn: $(OBJS)
	@$(LNKCMD) $(OBJS) $(LIBS) -o $(LIBIMF)
	@echo "===============  ::  "$(LIBIMF)" built"
	@echo "                    with flags " $(CFLAGS)
	@$(LNKCMDS) $(LIBIMFS) $(OBJS)
	@echo "========================  ::  "$(LIBIMFS)" built"
	@echo "                              with flags " $(CFLAGS)
	@echo "indexing the library..."
	@ranlib $(LIBIMFS)


$(OBJS): $(INCL)

%.o : %.c
	@echo "...............[$(CC)]" $< "->" $@
	@$(CC) -c $(CFLAGS) $< -o $@


clean:
	@echo "cleaning object files: "	
	@echo $(OBJS) $(LIBIMF)
	@rm -f $(OBJS) $(LIBIMF)
