KOKKOS_PATH ?= $(HOME)/kokkos/install
KOKKOS_INC = $(KOKKOS_PATH)/include
KOKKOS_LIB = $(KOKKOS_PATH)/lib64

CUDA_PATH ?= $(CUDA_DIR)
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

NVCC_WRAPPER = $(KOKKOS_PATH)/bin/nvcc_wrapper
OPTS = -O3

CHARM_PATH = $(HOME)/work/charm
CHARMC = $(CHARM_PATH)/bin/charmc -c++ $(NVCC_WRAPPER) $(OPTS)
CHARM_LD = $(CHARM_PATH)/bin/charmc $(OPTS)
CHARM_INC = $(HOME)/work/charm/include

TARGET = miniMD

all: $(TARGET)

$(TARGET): ljs.o input.o atom.o neighbor.o comm.o force_eam.o force_lj.o integrate.o thermo.o setup.o
	$(CHARM_LD) -language charm++ -L$(KOKKOS_LIB) -lkokkoscore -L$(CUDA_LIB) -o $@ $^

ljs.o: ljs.cpp $(TARGET).decl.h $(TARGET).def.h ljs_kokkos.h
	$(CHARMC) -I$(KOKKOS_INC) -c $<

$(TARGET).decl.h: ljs.ci
	$(CHARMC) $<

$(TARGET).def.h: ljs.ci

input.o: input.cpp
	$(CHARMC) -c $<

atom.o: atom.cpp atom.h
	$(CHARMC) -c -I$(KOKKOS_INC) $<

neighbor.o: neighbor.cpp neighbor.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

comm.o: comm.cpp comm.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

force_eam.o: force_eam.cpp force_eam.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

force_lj.o: force_lj.cpp force_lj.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

integrate.o: integrate.cpp integrate.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

thermo.o: thermo.cpp thermo.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

setup.o: setup.cpp
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun
