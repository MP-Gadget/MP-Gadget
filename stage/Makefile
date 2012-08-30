
OPT	+=  -DDOUBLEPRECISION

#SYSTYPE="HLRB2"
#SYSTYPE="MPA"
SYSTYPE="Regatta"
#SYSTYPE="OpteronMPA"
#SYSTYPE="OPA-Cluster64"


CC       = cc        # sets the C-compiler (default)
OPTIMIZE = -Wall  -g   # optimization and warning flags (default)


ifeq ($(SYSTYPE),"HLRB2")
CC       =  icc   # sets the C-compiler
OPTIMIZE =  -g -O3 -ipo  -mtune=itanium2 -mcpu=itanium2 -mp
endif


ifeq ($(SYSTYPE),"OPA-Cluster64")
CC       =   cc
endif


ifeq ($(SYSTYPE),"OpteronMPA")
CC       =  cc  -m64 # sets the C-compiler
OPTIMIZE =   -O3 -Wall
endif


ifeq ($(SYSTYPE),"MPA")
CC       =  cc   # sets the C-compiler
OPTIMIZE =   -g -Wall 
endif



ifeq ($(SYSTYPE),"Regatta")
CC       =  xlc_r -g # -qflttrap=enable:zerodivide:nanq # sets the C-compiler
OPTIMIZE =  -qstrict -q64 -qcpluscmt  -O 
endif


OPTIONS = $(OPTIMIZE) $(OPT) 

EXEC   = Stage

OBJS  =  subfind_stage.o mymalloc.o allvars.o read_parameters.o

INCL   = proto.h allvars.h Makefile

CFLAGS = $(OPTIONS) 

LIBS   = -lm 


$(EXEC): $(OBJS)
	$(CC) $(OPTIONS) $(OBJS) $(LIBS) $(RLIBS) -o $(EXEC)

$(OBJS): $(INCL)

clean:
	rm -f $(OBJS) $(EXEC)

