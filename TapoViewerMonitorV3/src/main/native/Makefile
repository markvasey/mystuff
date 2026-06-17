CC   = g++
# Locate nvcc from the CUDA 11.2 system package (runtime driver is 13.0 — compatible)
NVCC := $(shell which nvcc 2>/dev/null || find /usr/lib/cuda-*/bin/nvcc /usr/local/cuda/bin/nvcc -type f 2>/dev/null | head -n 1)
ifeq ($(NVCC),)
NVCC = nvcc
endif

# NPP library paths (CUDA 11.2 toolkit)
CUDA_LIB_DIR := $(shell dirname $(NVCC))/../targets/x86_64-linux/lib

# Priority 2: link NPP libraries needed for:
#   libnppc    — core (nppSetStream)
#   libnppisu  — statistics (nppiNormDiff_L1_8u_C1R, nppiNormDiffGetBufferHostSize)
#   libnppig   — geometry transforms (nppiResize_8u_C3R)
#   libnppidei — data exchange (nppiConvert_8u32f_C3R, nppiSwapChannels_8u_C3R)
#   libnppial  — arithmetic (nppiDivC_32f_C3IR)
#   libnppicc  — colour conversion (channel-order functions)
NPP_LIBS = -lnppc -lnppisu -lnppist -lnppig -lnppidei -lnppial -lnppicc

NVCCFLAGS = -O3 --shared -Xcompiler -fPIC -DUSE_CUDA -ccbin g++-9 \
            -L$(CUDA_LIB_DIR) $(NPP_LIBS)

TARGET = ../../../libseizure_cuda.so

.PHONY: all clean

all: $(TARGET)

$(TARGET): optical_flow.cu optical_flow.h
	$(NVCC) $(NVCCFLAGS) optical_flow.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
