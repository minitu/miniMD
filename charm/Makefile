KOKKOS_PATH ?= $(HOME)/kokkos/install
KOKKOS_INC = $(KOKKOS_PATH)/include
KOKKOS_LIB = $(KOKKOS_PATH)/lib64

CUDA_PATH ?= $(CUDA_DIR)
CUDA_INC = $(CUDA_PATH)/include
CUDA_LIB = $(CUDA_PATH)/lib64

NVCC_WRAPPER = $(KOKKOS_PATH)/bin/nvcc_wrapper
OPTS = -O3

CHARM_PATH = $(HOME)/charm
CHARMC = $(CHARM_PATH)/bin/charmc -c++ $(NVCC_WRAPPER) $(OPTS)
CHARM_LD = $(CHARM_PATH)/bin/charmc $(OPTS)
CHARM_INC = $(CHARM_PATH)/include

TARGET = miniMD

all: $(TARGET)

$(TARGET): ljs.o input.o atom.o neighbor.o comm.o force_eam.o force_lj.o integrate.o thermo.o setup.o block.o
	$(CHARM_LD) -language charm++ -L$(KOKKOS_LIB) -lkokkoscore -L$(CUDA_LIB) -o $@ $^

$(TARGET).decl.h: ljs.ci
	$(CHARMC) $<

$(TARGET).def.h: ljs.ci

test: test.ci
	$(CHARMC) $<

%.o: %.cpp $(TARGET).decl.h $(TARGET).def.h
	$(CHARMC) -c -I$(KOKKOS_INC) -I$(CHARM_INC) $<

clean:
	rm -f $(TARGET) *.decl.h *.def.h *.o charmrun

clean-nsys:
	rm -f *.qdrep *.qdstrm *.sqlite

clean-nvprof:
	rm -f *.nvvp
