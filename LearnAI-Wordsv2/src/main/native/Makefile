CC = g++
NVCC = nvcc
CFLAGS = -O3 -shared -fPIC -Wall -Wextra
NVCCFLAGS = -O3 --shared -Xcompiler -fPIC -DUSE_CUDA
TARGET = ../../../libwords_cuda.so

.PHONY: all mock cuda clean

all: mock

mock: words_cuda.cu words_cuda.h
	$(CC) $(CFLAGS) -x c++ words_cuda.cu -o $(TARGET)

cuda: words_cuda.cu words_cuda.h
	$(NVCC) $(NVCCFLAGS) words_cuda.cu -o $(TARGET)

clean:
	rm -f $(TARGET)
