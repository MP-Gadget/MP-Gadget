
#SYSTYPE="Regatta"
#SYSTYPE="RZG_LinuxCluster"
SYSTYPE="RZG_LinuxCluster-gcc"

CC       =   cc      
CFLAGS   =   -O2 -Wall -fPIC


LL       =  ld      
LFLAGS   =  -shared 
LIBS     =  -lm


ifeq ($(SYSTYPE),"Regatta")
CC       =  cc    # sets the C-compiler
CFLAGS   =  -O2  
LL       =  cc        
LFLAGS   =  -bM:SRE -bnoentry -bE:export.txt 
endif

ifeq ($(SYSTYPE),"RZG_LinuxCluster")
CC       =   icc   # sets the C-compiler
CFLAGS   =   -O2  -fPIC
endif

ifeq ($(SYSTYPE),"RZG_LinuxCluster-gcc")
CC       =   gcc   # sets the gcc C-compiler
OPTIMIZE =   -O2 -Wall -fPIC
endif



EXEC   = idlgrouplib.so

OBJS   = groupstuff.o 

INCL   = groupstuff.h


$(EXEC): $(OBJS) 
	$(LL) $(LFLAGS) $(OBJS) $(LIBS)  -o  $(EXEC)  

$(OBJS): $(INCL) 

clean:
	rm -f $(OBJS) $(EXEC)






