
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

FFLAGS = -132

NVCCINCLUDE = -I/home/wang/matlab/quantum-dynamics/common-cuda \
	-I$(CUDA_ROOT)/samples/common/inc \
	-I$(MATLAB_ROOT)/extern/include

CUDAArchs= -gencode arch=compute_20,code=sm_20 \
        -gencode arch=compute_30,code=sm_30 \
        -gencode arch=compute_35,code=sm_35 \
        -gencode arch=compute_37,code=sm_37 \
        -gencode arch=compute_50,code=sm_50 \
	-Xcompiler=\"-fPIC -pthread -fexceptions -m64\"

NVCCFLAGS = $(NVCCINCLUDE) $(CUDAArchs) -use_fast_math -O3 

CXXFLAGS = -std=c++0x $(NVCCINCLUDE)

Link = $(CXX)

LIBS = -lfftw3_omp -lifcoremt -L$(CUDA_LIB) -lcufft -lcublas -lcudart

MEXA64Files = \
	$(O)/TimeEvolutionMexCUDA.mexa64 \
	$(O)/GaussLegendreMex.mexa64

CUDAObjs = $(O)/harmonic.o $(O)/timeEvol2.o $(O)/cumath.o $(O)/evolutionCUDA2.o \
	$(O)/myreduction.o
#	$(O)/reduction_kernel.o

CUDALinkObj = $(O)/cudalink.o

OBJS = $(O)/GaussLegendreMex.o $(O)/GaussLegendre.o $(O)/sortcpp.o \
	$(O)/fftwinterface.o $(O)/timeEvolCUDA.o  $(O)/LegTransform.o \
	$(O)/zeros.o $(O)/CRPAux.o $(O)/wave2matlab.o \
	$(O)/Gradient3.o $(O)/matutils.o \
	$(O)/MatlabStructures.o $(O)/MatlabStructuresio.o $(O)/out.o \
	$(O)/evolutionCUDA.o \
	$(CUDAObjs) $(CUDALinkObj)

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
	cd $(O) ; $(NVCC) $(NVCCFLAGS) -dc $(SRC)/$<

$(CUDALinkObj): $(CUDAObjs)
	cd $(O); $(NVCC) $(CUDAArchs) -dlink $(CUDAObjs) -o $(CUDALinkObj)

%io.C: %.h
	perl io.pl $<

$(QMLibs): $(OBJS)
	cd $(O); ar -crusv $(QMLibs) $(OBJS)

$(O)/%.mexa64: $(O)/%.o $(QMLibs)
	cd $(O); $(Link) -shared $(CXXFLAGS) -o $@ $^ $(LIBS)

clean:
	rm -f $(O)/* *.o *~ *.mod $(EXE) depend $(MEXA64Files) $(QMLibs) $(OBJS)

cleancuda:
	rm -rf $(CUDAObjs)

depend :
	$(CXX) $(CXXFLAGS) -MM *.[cC] | perl dep.pl > $@
	$(NVCC) $(NVCCINCLUDE) -M *.cu | perl dep.pl >> $@

include depend
