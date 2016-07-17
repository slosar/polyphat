#include "settings.h"
#define _XOPEN_SOURCE 700
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "polyphat.h"

#define DO1 1
#define DOOUT 0
#define CLOCKP  1000000000L;
#define NTEST (400000000/_PPNTOT_)
//#20000

int main() {
  printf("Allocating and sampling\n");
  float* sam=polyphase_alloc_buffer(_PPNTOT_);
  float frq[_PPNF_], pwr[_PPNF_];
  double dt=1.0/2.5e9; /*2.5 GSample/s;*/
  for (float f=1.; f<1215.; f+=101.) {
    printf ("Adding %f\n",f);
    double ofs=(double)f;
    for (int i=0;i<_PPNTOT_; i++) {
      double phi=ofs+i*dt*2*M_PI*f*1e6;
      if (f<10) sam[i]=sin(phi); else sam[i]+=sin(phi);
    }
  }
  
  printf ("Charging FFT\n");
  polyphase_charge();

  struct timespec requestStart, requestEnd;
  double accum;


    printf ("Polyphasing \n");
    // Calculate time taken by a request
    clock_gettime(CLOCK_REALTIME, &requestStart);
    for (size_t cc=0; cc<NTEST; cc++) {
        if (DO1) polyPhase_cuda(sam, pwr);
    }
   clock_gettime(CLOCK_REALTIME, &requestEnd);
    // Calculate time it took
    accum = ( requestEnd.tv_sec - requestStart.tv_sec )
      + (double)( requestEnd.tv_nsec - requestStart.tv_nsec )
      / CLOCKP;
    accum/=NTEST;
    printf( "Time: %g, rate %g MS/s for %i %i over %i tests. XX\n", accum,
	    _PPNTOT_/accum/1e6,_PPNS_, _PPNC_,NTEST);
  
    if (DOOUT) {
      printf ("Output\n");
      getFreqArray(dt,frq);
      for (int i=0; i<_PPNF_; i++)
	printf ("%g %g \n",frq[i]/1e6,pwr[i]);
    }
    polyphase_release();
  return 0;
}
