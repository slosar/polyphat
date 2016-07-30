
#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>
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


void print_timing (cudaEvent_t* start, cudaEvent_t* stop, char* what) {
  float gpu_time;
  CHK(cudaEventElapsedTime(&gpu_time, *start, *stop));
  printf ("Timing %s : %fms \n",what, gpu_time);
}


#define FLOATIZE_X 8
/**
 * CUDA Kernel byte->float
 *
 */
__global__ void floatize(uint8_t* sample,cufftReal* fsample)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]-128);
}




void cuda_test(uint8_t *buf) {

  // cuda buffer and float buffer
  uint8_t *cbuf;
  CHK(cudaMalloc(&cbuf,BUFFER_SIZE));
  cufftReal *cfbuf;
  CHK(cudaMalloc(&cfbuf,BUFFER_SIZE*sizeof(cufftReal)));
  cufftComplex *ffts;
  CHK(cudaMalloc(&ffts,TRANSFORM_SIZE*NUM_FFT*sizeof(cufftComplex)));

  cufftHandle plan;
  int oembed=TRANSFORM_SIZE*NUM_FFT+1;
  int fftsize=FFT_SIZE;
  int status=cufftPlanMany(&plan, 1, &fftsize, NULL, 0, 0, 
        NULL, TRANSFORM_SIZE,1, CUFFT_R2C, NUM_FFT);
  if (status!=CUFFT_SUCCESS) {
       printf ("Plan failed:");
       if (status==CUFFT_ALLOC_FAILED) printf("CUFFT_ALLOC_FAILED");
       if (status==CUFFT_INVALID_VALUE) printf ("CUFFT_INVALID_VALUE");
       if (status==CUFFT_INTERNAL_ERROR) printf ("CUFFT_INTERNAL_ERROR");
       if (status==CUFFT_SETUP_FAILED) printf ("CUFFT_SETUP_FAILED");
       if (status==CUFFT_INVALID_SIZE) printf ("CUFFT_INVALID_SIZE");
       printf("\n");
       exit(1);
  }

  cudaEvent_t tstart, tcpy,tfloatize,tfft;
  CHK(cudaEventCreate(&tstart));
  CHK(cudaEventCreate(&tcpy));
  CHK(cudaEventCreate(&tfloatize));
  CHK(cudaEventCreate(&tfft));

  cudaEventRecord(tstart, 0);
  // copy to device
  CHK(cudaMemcpy(cbuf,buf, BUFFER_SIZE, cudaMemcpyHostToDevice));

  cudaEventRecord(tcpy, 0);
  
  // floatize
  int threadsPerBlock = 1024;
  int blocksPerGrid = BUFFER_SIZE / threadsPerBlock/FLOATIZE_X;
  floatize<<<blocksPerGrid, threadsPerBlock>>>(cbuf,cfbuf);
  cudaEventRecord(tfloatize, 0);
  CHK(cudaGetLastError());
  
  status=cufftExecR2C(plan, cfbuf, ffts);
  cudaEventRecord(tfft, 0);
  if (status!=CUFFT_SUCCESS) {
     printf("CUFFT FAILED\n");
     exit(1);
  }    

  cudaThreadSynchronize();
  print_timing(&tstart,&tcpy,"MEM CPY");
  print_timing(&tcpy,&tfloatize,"FLOATIZE");
  print_timing(&tfloatize,&tfft,"FFT");

}


void ztest() {
  uint8_t *cbuf;
  CHK(cudaMalloc(&cbuf,BUFFER_SIZE));
  // floatize
  int threadsPerBlock = 1024;
  int blocksPerGrid = 32768;
  int Nth=
  printf ("%i %i",threadsPerBlock, blocksPerGrid);
  //  floatize<<<blocksPerGrid, threadsPerBlock>>>(cbuf);
  CHK(cudaGetLastError());

}
