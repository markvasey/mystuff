CC = g++
# Dynamically locate nvcc if it's not in the PATH
NVCC := $(shell which nvcc 2>/dev/null || find /usr/lib/cuda-*/bin/nvcc /usr/local/cuda/bin/nvcc -type f 2>/dev/null | head -n 1)
ifeq ($(NVCC),)
NVCC = nvcc
endif
NVCCFLAGS = -O3 --shared -Xcompiler -fPIC -DUSE_CUDA -lcublas -ccbin g++-9
TARGET = ../../../libwords_cuda.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): words_cuda.cu words_cuda.h
	$(NVCC) $(NVCCFLAGS) words_cuda.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
