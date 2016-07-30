
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "settings.h"

extern "C" {
#include "processor_cuda.h"
}

#include <stdio.h>
#include "math.h"

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define CHK( err ) (HandleError( err, __FILE__, __LINE__ ))


uint8_t* alloc_sample_buffer() {
  uint8_t *p;
  /* size of uint8_t is one, explicityly */
  CHK(cudaHostAlloc(&p, BUFFER_SIZE, cudaHostAllocDefault));
 return p;
}



/**
 * CUDA Kernel byte->float
 *
 */
__global__ void floatize(uint8_t *sample, float* fsample)  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<BUFFER_SIZE) fsample[i]=float(sample[i]-128);
}


void print_timing (cudaEvent_t start, cudaEvent_t stop, char* what) {
  float gpu_time;
  cudaEventElapsedTime(&gpu_time, start, stop);
  printf ("Timing %s : %fms ",what, gpu_time);
}



void cuda_test(uint8_t *buf) {

  // cuda buffer and float buffer
  uint8_t *cbuf;
  CHK(cudaMalloc(&cbuf,BUFFER_SIZE));
  float *cfbuf;
  CHK(cudaMalloc(&cfbuf,BUFFER_SIZE*sizeof(float)));

  cudaEvent_t tstart, tcpy,tfloatize;
  CHK(cudaEventCreate(&tstart));
  CHK(cudaEventCreate(&tcpy));
  CHK(cudaEventCreate(&tfloatize));

  cudaEventRecord(tstart, 0);
  // copy to device
  cudaMemcpy(cbuf,buf, BUFFER_SIZE, cudaMemcpyHostToDevice);
  // floatize
  cudaEventRecord(tcpy, 0);

  int threadsPerBlock = 256;
  int blocksPerGrid = BUFFER_SIZE / threadsPerBlock;
  
  floatize<<<blocksPerGrid, threadsPerBlock>>>(cbuf, cfbuf);
  CHK(cudaGetLastError());
  cudaEventRecord(tfloatize, 0);


}

