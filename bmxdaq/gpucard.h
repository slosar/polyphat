/************e***********************
***********************************
THIS IS A COMPLETE PLACEHOLDER!
***********************************
**********************************/
#pragma once



#include "stdint.h"
#include "settings.h"


struct GPUCARD {
  void **cbuf; // pointer to pointers of GPU sample buffer
  void **cfbuf; // floats
  void **cfft; // ffts
  int nchan; // nchannels
  uint32_t fftsize; // fft size
  uint32_t bufsize; // buffer size in bytes
  int threads; // threads to use
  int plan;
  int nstreams;
  void *streams; // streams
  int fstream, bstream; // front stream (oldest running), back stream (newest runnig);
  int active_streams; // really needed just at the beginning (when 0)
  void *eStart, *eDoneCopy, *eDoneFloatize, *eDoneFFT, *eDonePost; //events

  
};


extern "C" {
  void gpuCardInit (GPUCARD *gcard, SETTINGS *set);
  bool gpuProcessBuffer(GPUCARD *gcard, int8_t *buf);
}
