/*********************

Polyphase filter implementation.

When compiling, need to define 
_PPNS_ -- number of samples in chunk
_PPNC_  -- number of chunks in total sample

WARNING: code assumes both numbers are even

***********************/

#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "settings.h"
extern "C" {
#include "polyphat_cuda.h"
}
#include <stdio.h>

#define M_PI 3.14159265358979323846

float* _cuda_filter_;
float* _cuda_sample_;
float* _cuda_output_;

float* polyphase_alloc_buffer(int size) {
  cudaError_t err = cudaSuccess;
  float *p;     
  err=cudaHostAlloc(&p, size*sizeof(float),
                            cudaHostAllocDefault);
  if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate host buffer  vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
return p;
}

void cuda_alloc(float *filter) {
  cudaError_t err = cudaSuccess;
  cudaMalloc(&_cuda_filter_,sizeof(float)*_PPNTOT_);
  if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector filter (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  err=cudaMalloc(&_cuda_sample_,sizeof(float)*_PPNTOT_);
  if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector sample (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
  err=cudaMalloc(&_cuda_output_,sizeof(float)*_PPNS_);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device vector output (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
    }

  err = cudaMemcpy(_cuda_filter_,filter, _PPNTOT_*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector filter from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

};


/**
 * CUDA Kernel Device code
 *
 */
__global__ void polyphase_do(float* sample, float* filter, float* output)  {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j=i;
    float sum=0.0;
    while (j<_PPNTOT_) {
      sum+=sample[j]*filter[j];
      j+=_PPNS_;
    }
    output[i]=sum;
}


void cuda_exec(float*sam, float* out) {
  cudaError_t err = cudaSuccess;
  err = cudaMemcpy(_cuda_sample_,sam, _PPNTOT_*sizeof(float), cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector data from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  int threadsPerBlock = 256;
  int blocksPerGrid =_PPNS_ / threadsPerBlock;
  
#ifndef NO_WORK_JUST_TRANSFER
  polyphase_do<<<blocksPerGrid, threadsPerBlock>>>(_cuda_sample_, _cuda_filter_, _cuda_output_);
  err = cudaGetLastError();
#endif
if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch cuda kernel (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(out,_cuda_output_, _PPNS_*sizeof(float), cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}


void cuda_release() {
  cudaFree(_cuda_filter_);
  cudaFree(_cuda_sample_);
  cudaFree(_cuda_output_);
}



