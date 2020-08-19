KOKKOS_PATH ?= $(HOME)/kokkos/install
KOKKOS_INC = $(KOKKOS_PATH)/include
KOKKOS_LIB = $(KOKKOS_PATH)/lib64

CUDA_PATH ?= $(CUDA_DIR)
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

NVCC_WRAPPER = $(KOKKOS_PATH)/bin/nvcc_wrapper
NVCC_OPTS = -O3 -std=c++11 -fopenmp

CHARM_PATH = $(HOME)/work/charm
CHARMC = $(CHARM_PATH)/bin/charmc $(OPTS)
CHARM_INC = $(HOME)/work/charm/include

TARGET = miniMD

all: $(TARGET)

$(TARGET): ljs.o ljs_kokkos.o input.o atom.o neighbor.o
	$(CHARMC) -language charm++ -fopenmp -L$(KOKKOS_LIB) -lkokkoscore -L$(CUDA_LIB) -o $@ $^

ljs.o: ljs.cpp $(TARGET).decl.h $(TARGET).def.h ljs_kokkos.h
	$(CHARMC) -I$(KOKKOS_INC) -fopenmp -c $<

ljs_kokkos.o: ljs_kokkos.cpp ljs_kokkos.h
	$(NVCC_WRAPPER) $(NVCC_OPTS) -c -I$(KOKKOS_INC) $<

$(TARGET).decl.h: ljs.ci
	$(CHARMC) $<

$(TARGET).def.h: ljs.ci

input.o: input.cpp
	$(CHARMC) -c $<

atom.o: atom.cpp atom.h
	$(NVCC_WRAPPER) $(NVCC_OPTS) -c -I$(KOKKOS_INC) $<

neighbor.o: neighbor.cpp neighbor.h
	$(NVCC_WRAPPER) $(NVCC_OPTS) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun
