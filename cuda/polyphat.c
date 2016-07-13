/*********************

Polyphase filter implementation.

When compiling, need to define 
_PPNS_ -- number of samples in chunk
_PPNC_  -- number of chunks in total sample

WARNING: code assumes both numbers are even

***********************/

#include <math.h>
#include <stdint.h>
#include <fftw3.h>
#include "settings.h"
#include "polyphat_cuda.h"

#define M_PI 3.14159265358979323846

void getFreqArray (float dt, float* arr) {
  unsigned int Nf=_PPNF_;
  float tmax=_PPNS_*dt;
  for (int i=0;i<Nf;i++) {
    arr[i] = i/tmax;
  }
}

fftwf_plan _plan_;
float* _fft_input_;
fftwf_complex* _output_;
float _filter_[_PPNTOT_];


void polyphase_charge() {
  _fft_input_ = fftwf_malloc(sizeof(float) * _PPNS_);
  _output_ = fftwf_malloc(sizeof(fftwf_complex) * _PPNF_);
  _plan_=fftwf_plan_dft_r2c_1d(_PPNS_, _fft_input_, _output_,FFTW_MEASURE);
  
  for (int i=0;i<_PPNTOT_;i++) {
    double t=(i-(_PPNTOT_/2))/(double)(_PPNS_)*M_PI;
    if (t==0) t=1.0; else t=sin(t)/t;
    t*=0.5*(1-cos(i*2*M_PI/_PPNTOT_));
    _filter_[i]=t;
  }
  cuda_alloc(_filter_);
};

void polyphase_release() {
  fftwf_destroy_plan(_plan_);
  fftwf_free(_fft_input_);
  fftwf_free(_output_);
  cuda_release();
}



void polyPhase_cuda(float *sam, float *spec) {
  cuda_exec(sam,_fft_input_);
  fftwf_execute(_plan_);
  for (unsigned int i=0;i<_PPNF_;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}

