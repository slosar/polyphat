#
# Settings mak include things that you might need to make it work,
# eg gcc and NVCC settings
#
-include settings.mak


FFTWLINK=-lfftw3f

all: make_test_buffers.out test.out

.c.o: settings.h
	$(GCC) -Ofast -std=c99 -c -o $@ $<

processor_cuda.o: processor_cuda.cu processor_cuda.h settings.h
	$(NVCC)  -O3 -c -o $@ $<

make_test_buffers.out: make_test_buffers.o generator.o processor_cuda.o
	$(NVCC) -o make_test_buffers.out make_test_buffers.o generator.o processor_cuda.o -lm 

test.out: test.o generator.o processor_cuda.o
	$(NVCC) -o test.out test.o generator.o processor_cuda.o -lm 

clean:
	rm *.o
