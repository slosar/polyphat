#
# Settings mak include things that you might need to make it work,
# eg gcc and NVCC settings
#
GCC=gcc
NVCC=nvcc
-include settings.mak


FFTWLINK=-lfftw3f

all: test.out

test.o: test.c settings.h
	$(GCC) -Ofast -std=c99 -c test.c

polyphat.o: polyphat.c settings.h
	$(GCC) -Ofast -std=c99 -c polyphat.c

polyphat_cuda.o: polyphat_cuda.cu settings.h
	$(NVCC)  -O3 -c polyphat_cuda.cu 

test.out: test.o polyphat.o polyphat_cuda.o 
	$(NVCC) -o test.out test.o polyphat.o polyphat_cuda.o -lm $(FFTWLINK)

clean:
	rm *.o
