KOKKOS_PATH ?= $(HOME)/kokkos/install
KOKKOS_INC = $(KOKKOS_PATH)/include
KOKKOS_LIB = $(KOKKOS_PATH)/lib

NVCC_WRAPPER = $(KOKKOS_PATH)/bin/nvcc_wrapper
NVCC_OPTS = -O3

CHARM_PATH = $(HOME)/work/charm
CHARMC = $(CHARM_PATH)/bin/charmc $(OPTS)

TARGET = miniMD
EXE = $(TARGET).charm

all: $(TARGET)

$(TARGET): $(TARGET).o
	$(CHARMC) -language charm++ -fopenmp -L$(KOKKOS_LIB) -lkokkos -o $@ $^

$(TARGET).o: ljs.cpp $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -I$(KOKKOS_INC) -fopenmp -c $<

$(TARGET).decl.h: ljs.ci
	$(CHARMC) $<

$(TARGET).def.h: ljs.ci

clean:
	rm -f $(EXE) *.decl.h *.def.h *.o charmrun
