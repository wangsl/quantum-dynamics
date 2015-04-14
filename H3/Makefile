
# $Id$

.DEFAULT: .F .F90 .c .C
.SUFFIXES: .F .F90 .c .C

SRC = /home/wang/matlab/quantum-dynamics/H3
O = /home/wang/matlab/quantum-dynamics/build/H3

F77 = ifort
F90 = ifort
CC = icc
CXX = icpc

CFLAGS =  

CXXFLAGS = -std=c++0x -I$(MATLAB_ROOT)/extern/include -I/home/wang/matlab/quantum-dynamics/common

FFLAGS = -132

Link = $(CXX)

LIBS = -lifcoremt

MEXA64Files = $(O)/BKMP2Mex.mexa64 

OBJS = $(O)/bkmp2.o $(O)/matutils.o 

QMLibs = $(O)/libqmdyn.a

.DEFAULT_GOAL := $(O)/BKMP2Mex.mexa64 

all: $(MEXA64Files)

#$(EXE) : $(OBJS)
#	$(Link) $(CXXFLAGS) -o $(EXE) $(OBJS) $(LIBS)

$(O)/%.o: %.c
	cd $(O) ; $(CC)  $(cFLAGS) -c $(SRC)/$<
$(O)/%.o: %.C
	cd $(O) ; $(CXX) $(CXXFLAGS) -c $(SRC)/$<
$(O)/%.o: %.F
	cd $(O) ; $(F77) $(FFLAGS) -c $(SRC)/$<
$(O)/%.o: %.F90
	cd $(O) ; $(F90) $(FFLAGS) -c $(SRC)/$<
%io.C: %.h
	perl io.pl $<

$(QMLibs): $(OBJS)
	cd $(O); ar -crusv $(QMLibs) $(OBJS)

$(O)/%.mexa64: $(O)/%.o $(QMLibs)
	cd $(O); $(Link) -shared $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o *~ *.mod $(EXE) depend $(MEXA64Files) $(QMLibs) $(OBJS)

depend :
	$(CXX) $(CXXFLAGS) -MM *.[cC] | perl dep.pl | tee $@

include depend
