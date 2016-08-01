#
# Settings mak include things that you might need to make it work,
# eg gcc and NVCC settings, if you want to override
#
GCC=gcc
NVCC=nvcc 
-include settings.mak
FFTWLINK=-lfftw3f

SOURCES=generator.c processor_cuda.c settings.c
CUDASOURCES=processor_cuda.cu
OBJS=$(SOURCES:.c=.o) $(CUDASOURCES:.cu=.o) 

all: make_test_buffers.out test.out

%.o: %.c settings.h
	$(GCC) -Ofast -std=c99 -c -o $@ $<

%.o: %.cu settings.h
	$(NVCC)  -O3 -c -o $@ $<

libfftproc.a: $(OBJS)
	ar rcs libfftproc.a $(OBJS)

make_test_buffers.out: make_test_buffers.o libfftproc.a
	$(NVCC) -o make_test_buffers.out make_test_buffers.o -L. -lfftproc -lm -lcufft

test.out: test.o generator.o processor_cuda.o
	$(NVCC) -o test.out test.o -L. -lfftproc -lm -lcufft

clean:
	rm *.o
