NVCC = /usr/local/cuda/bin/nvcc 
NVCC_FLAGS = -g -G -Xcompiler -Wall -dc

all: decoding

decoding: decoding.o aux.o llrv.o
	$(NVCC) $^ -o $@

decoding.o: decoding.cu aux.h llrv.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

aux.o: aux.cpp aux.h
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

llrv.o: llrv.cu llrv.cuh gfcalu.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
.PHONY:clean
clean:
	rm decoding decoding.o aux.o