
# $Id$

.DEFAULT: .F .F90 .c .C
.SUFFIXES: .F .F90 .c .C

SRC = /home/wang/matlab/quantum-dynamics/common-cuda
O = /home/wang/matlab/quantum-dynamics/build/common-cuda

F77 = ifort
F90 = ifort
CC = icc
CXX = icpc
NVCC = nvcc

CFLAGS =  

CXXFLAGS = -std=c++0x -I$(MATLAB_ROOT)/extern/include

FFLAGS = -132

NVCCINCLUDE = -I$(CUDA_ROOT)/samples/common/inc

NVCCFLAGS = \
        -gencode arch=compute_20,code=sm_20 \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_37,code=sm_37 \
        -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_52,code=sm_52 \
        -gencode arch=compute_52,code=compute_52 \
        -use_fast_math -O3 \
        $(NVCCINCLUDE) $(TYPEDEF) -Xcompiler=\"-fPIC -pthread -fexceptions -m64\"

Link = $(CXX)

LIBS = -lfftw3_omp -lifcoremt -L$(CUDA_LIB) -lcufft -lcublas -lcudart 

MEXA64Files = \
	$(O)/TimeEvolutionMexCUDA.mexa64 \
	$(O)/GaussLegendreMex.mexa64

OBJS = $(O)/GaussLegendreMex.o $(O)/GaussLegendre.o $(O)/sortcpp.o \
	$(O)/fftwinterface.o $(O)/timeEvolCUDA.o  $(O)/LegTransform.o \
	$(O)/zeros.o $(O)/CRPAux.o $(O)/wave2matlab.o \
	$(O)/Gradient3.o $(O)/matutils.o \
	$(O)/MatlabStructures.o $(O)/MatlabStructuresio.o $(O)/out.o \
	$(O)/harmonic.o

QMLibs = $(O)/libqmdyn.a

.DEFAULT_GOAL := $(O)/TimeEvolutionMexCUDA.mexa64

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
$(O)/%.o: %.cu
	cd $(O) ; $(NVCC) $(NVCCFLAGS) -c $(SRC)/$<

%io.C: %.h
	perl io.pl $<

$(QMLibs): $(OBJS)
	cd $(O); ar -crusv $(QMLibs) $(OBJS)

$(O)/%.mexa64: $(O)/%.o $(QMLibs)
	cd $(O); $(Link) -shared $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f *.o *~ *.mod $(EXE) depend $(MEXA64Files) $(QMLibs) $(OBJS)

depend :
	$(CXX) $(CXXFLAGS) -MM *.[cC] | perl dep.pl > $@
	$(NVCC) $(NVCCINCLUDE) -M *.cu | perl dep.pl >> $@

include depend
