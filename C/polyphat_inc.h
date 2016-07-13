/*********************

Polyphase filter implementation.

When compiling, need to define 
_PPNS_ -- number of samples in chunk
_PPNC_  -- number of chunks in total sample

WARNING: code assumes both numbers are even

***********************/

#include <math.h>
#include <stdint.h>
#include "fftw3.h"
#define M_PI 3.14159265358979323846
#define _PPNTOT_ (_PPNS_*_PPNC_)
#define _PPNF_ (_PPNS_/2+1)

void getFreqArray (float dt, float* arr) {
  unsigned int Nf=_PPNF_;
  float tmax=_PPNS_*dt;
  for (int i=0;i<Nf;i++) {
    arr[i] = i/tmax;
  }
}

fftwf_plan _plan_;
float* _input_;
fftwf_complex* _output_;
float _filter_[_PPNTOT_];

void chargeFFT() {
  _input_ = fftwf_malloc(sizeof(float) * _PPNS_);
  _output_ = fftwf_malloc(sizeof(fftwf_complex) * _PPNF_);
  _plan_=fftwf_plan_dft_r2c_1d(_PPNS_, _input_, _output_,FFTW_MEASURE);
  
  for (int i=0;i<_PPNTOT_;i++) {
    double t=(i-(_PPNTOT_/2))/(double)(_PPNS_)*M_PI;
    if (t==0) t=1.0; else t=sin(t)/t;
    t*=0.5*(1-cos(i*2*M_PI/_PPNTOT_));
    _filter_[i]=t;
  }
};

void releaseFFT() {
  fftwf_destroy_plan(_plan_);
  fftwf_free(_input_);
  fftwf_free(_output_);
}

void polyPhase(float *sam, float *spec) {
  for (int k=0;k<_PPNS_;k++) _input_[k]=0.0;
  unsigned int k=0;
  for (int i=0;i<_PPNTOT_;i++) {
    float t=(i-(_PPNTOT_/2))/(double)(_PPNS_)*M_PI;
    if (t==0) t=1.0; else t=sin(t)/t;
    t*=0.5*(1-cos(i*2*M_PI/_PPNTOT_));
    _input_[k]+=sam[i]*t;
    k+=1; if (k==_PPNS_) k=0;
  }
  fftwf_execute(_plan_);
  unsigned int Nf=_PPNF_;
  for (unsigned int i=0;i<Nf;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}

void polyPhase_2(float *sam, float *spec) {
  for (int k=0;k<_PPNS_;k++) _input_[k]=0.0;
  unsigned int k=0;
  for (int i=0;i<_PPNTOT_;i++) {
    _input_[k]+=sam[i]*_filter_[i];
    k+=1; if (k==_PPNS_) k=0;
  }
  fftwf_execute(_plan_);
  unsigned int Nf=_PPNF_;
  for (unsigned int i=0;i<Nf;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}

void polyPhase_3(float *sam, float *spec) {
  for (int k=0;k<_PPNS_;k++) _input_[k]=0.0;
  for (int i=0;i<_PPNTOT_;i++) 
    _input_[i%_PPNS_]+=sam[i]*_filter_[i];
  fftwf_execute(_plan_);
  unsigned int Nf=_PPNF_;
  for (unsigned int i=0;i<Nf;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}

void polyPhase_4(float *sam, float *spec) {
  for (int k=0;k<_PPNS_;k++) _input_[k]=0.0;
  for (int i=0;i<_PPNS_;i++)
   for (int j=i; j<_PPNTOT_; j+=_PPNS_)
     _input_[i]+=sam[j]*_filter_[j];
  fftwf_execute(_plan_);
  unsigned int Nf=_PPNF_;
  for (unsigned int i=0;i<Nf;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}

void polyPhase_5(float *sam, float *spec) {
  for (int k=0;k<_PPNS_;k++) _input_[k]=0.0;

  #pragma omp parallel for
  for (int i=0;i<_PPNS_;i++)
    for (int j=i; j<_PPNTOT_; j+=_PPNS_)
      _input_[i]+=sam[j]*_filter_[j];
  fftwf_execute(_plan_);
  unsigned int Nf=_PPNF_;
  for (unsigned int i=0;i<Nf;i++) {
    spec[i]=_output_[i][0]*_output_[i][0]+_output_[i][1]*_output_[i][1];
  }
}


