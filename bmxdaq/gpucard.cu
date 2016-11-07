/***********************************
***********************************
CUDA PART
***********************************
**********************************/

#include "gpucard.h"

#include <memory.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <stdio.h>
#include "math.h"

#define FLOATIZE_X 8


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
      printf( "CUDA fail: %s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit(1);
    }
}
#define CHK( err ) (HandleError( err, __FILE__, __LINE__ ))

void gpuCardInit (GPUCARD *gc, SETTINGS *set) {
  printf ("\n\nInitilizing GPU\n");
  printf ("=================\n");
  printf ("Allocating GPU buffers\n");
  int Nb=set->buf_mult;
  gc->cbuf=(void**)malloc(Nb*sizeof(void*));
  gc->cfbuf=(void**)malloc(Nb*sizeof(void*));
  gc->cfft=(void**)malloc(Nb*sizeof(void*));
  int nchan=gc->nchan=1+(set->channel_mask==3);
  if ((nchan==2) and (FLOATIZE_X%2==1)) {
    printf ("Need FLOATIZE_X even for two channels\n");
    exit(1);
  }
  gc->fftsize=set->fft_size;
  uint32_t bufsize=gc->bufsize=set->fft_size*nchan;
  uint32_t transform_size=(set->fft_size/2+1)*nchan;
  for (int i=0;i<Nb;i++) {
    uint8_t** cbuf=(uint8_t**)&(gc->cbuf[i]);
    CHK(cudaMalloc(cbuf,bufsize));
    cufftReal** cfbuf=(cufftReal**)&(gc->cfbuf[i]);
    CHK(cudaMalloc(cfbuf, bufsize*sizeof(cufftReal)));
    cufftComplex** ffts=(cufftComplex**)&(gc->cfft[i]);
    CHK(cudaMalloc(ffts,transform_size*sizeof(cufftComplex)));
  }


  printf ("Setting up CUFFT");
  int status=cufftPlanMany(&gc->plan, 1, (int*)&(set->fft_size), NULL, 0, 0, 
        NULL, transform_size,1, CUFFT_R2C, nchan);

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
  printf ("Setting up CUDA streams & events\n");
  gc->nstreams=set->cuda_streams;
  if (gc->nstreams<1) {
    printf ("Cannot relly work with less than one stream.\n");
    exit(1);
  }
  gc->streams=malloc(gc->nstreams*sizeof(cudaStream_t));

  gc->eStart=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneCopy=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFloatize=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDoneFFT=malloc(gc->nstreams*sizeof(cudaEvent_t));
  gc->eDonePost=malloc(gc->nstreams*sizeof(cudaEvent_t));
  cudaEvent_t* eStart=(cudaEvent_t*)(gc->eStart);
  cudaEvent_t* eDoneCopy=(cudaEvent_t*)(gc->eDoneCopy);
  cudaEvent_t* eDoneFloatize=(cudaEvent_t*)(gc->eDoneFloatize);
  cudaEvent_t* eDoneFFT=(cudaEvent_t*)(gc->eDoneFFT);
  cudaEvent_t* eDonePost=(cudaEvent_t*)(gc->eDonePost);

  for (int i=0;i<gc->nstreams;i++) {
    CHK(cudaEventCreate(&eStart[i]));
    CHK(cudaEventCreate(&eDoneCopy[i]));
    CHK(cudaEventCreate(&eDoneFloatize[i]));
    CHK(cudaEventCreate(&eDoneFFT[i]));
    CHK(cudaEventCreate(&eDonePost[i]));
  }
  gc->fstream=gc->bstream=gc->active_streams=0;
}




/**
 * CUDA Kernel byte->float, 1 channel version
 *
 */
__global__ void floatize_1chan(uint8_t* sample, cufftReal* fsample)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X; j++) fsample[i+j]=float(sample[i+j]);
}

__global__ void floatize_2chan(uint8_t* sample, cufftReal* fsample1, cufftReal* fsample2)  {
    int i = FLOATIZE_X*(blockDim.x * blockIdx.x + threadIdx.x);
    for (int j=0; j<FLOATIZE_X/2; j++) {
      fsample1[i+j]=float(sample[i+2*j]);
      fsample2[i+j]=float(sample[i+2*j+1]);
    }
}


bool gpuProcessBuffer(GPUCARD *gc, int8_t *buf) {

  // pointers and vars
  uint8_t** cbuf=(uint8_t**)(gc->cbuf);
  cufftReal** cfbuf=(cufftReal**)(gc->cfbuf);
  cufftComplex** cfft=(cufftComplex**)(gc->cfft);

  cudaEvent_t* eStart=(cudaEvent_t*)(gc->eStart);
  cudaEvent_t* eDoneCopy=(cudaEvent_t*)(gc->eDoneCopy);
  cudaEvent_t* eDoneFloatize=(cudaEvent_t*)(gc->eDoneFloatize);
  cudaEvent_t* eDoneFFT=(cudaEvent_t*)(gc->eDoneFFT);
  cudaEvent_t* eDonePost=(cudaEvent_t*)(gc->eDonePost);
  cudaStream_t* streams=(cudaStream_t*)gc->streams;

  // first check if there are buffers to store
  while (gc->active_streams>0) {
    // process done streams
    // IMPLEMENT
  }
  // add a new stream
  gc->active_streams++;
  int csi=gc->bstream = (++gc->bstream)%(gc->nstreams);
  cudaStream_t cs= streams[gc->bstream];
  cudaEventRecord(eStart[csi], cs);
  CHK(cudaMemcpyAsync(cbuf[csi], buf, gc->bufsize , cudaMemcpyHostToDevice,cs));
  cudaEventRecord(eDoneCopy[csi], cs);
  int threadsPerBlock = gc->threads;
  int blocksPerGrid = gc->bufsize / threadsPerBlock/FLOATIZE_X;
  if (gc->nchan==1) 
    floatize_1chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(cbuf[csi],cfbuf[csi]);
  else 
    floatize_2chan<<<blocksPerGrid, threadsPerBlock, 0, cs>>>(cbuf[csi],cfbuf[csi],&(cfbuf[csi][gc->fftsize]));
  cudaEventRecord(eDoneFloatize[csi], cs);
  int status=cufftExecR2C(gc->plan, cfbuf[csi], cfft[csi]);
  
  cudaEventRecord(eDoneFFT[csi], cs);
  cudaEventRecord(eDonePost[csi], cs);




  
  return true;
}
