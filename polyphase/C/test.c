#ifndef _PPNS_
#define _PPNS_ 2048
#endif
#ifndef _PPNC_
#define _PPNC_ 2048
#endif

#define _XOPEN_SOURCE 700

#define DO1 0
#define DO2 0
#define DO3 1
#define DO4 0
#define DO5 0

#define DOOUT 0

#include <time.h>
#define CLOCKP  1000000000L;
#define NTEST 200
#include "polyphat_inc.h"
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

int main() {
  printf("Allocating and sampling\n");
  float sam[_PPNTOT_];
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
  chargeFFT();

  struct timespec requestStart, requestEnd;
  double accum;


    printf ("Polyphasing \n");
    // Calculate time taken by a request
    clock_gettime(CLOCK_REALTIME, &requestStart);
    for (size_t cc=0; cc<NTEST; cc++) {
        if (DO1) polyPhase(sam, pwr);
        if (DO2) polyPhase_2(sam, pwr);
        if (DO3) polyPhase_3(sam, pwr);
	if (DO4) polyPhase_4(sam, pwr);
	if (DO5) polyPhase_5(sam, pwr);
    }
   clock_gettime(CLOCK_REALTIME, &requestEnd);
    // Calculate time it took
    accum = ( requestEnd.tv_sec - requestStart.tv_sec )
      + (double)( requestEnd.tv_nsec - requestStart.tv_nsec )
      / CLOCKP;
    accum/=NTEST;
    printf( "Time: %g, rate %g MS/s for %i %i XX\n", accum, _PPNTOT_/accum/1e6,_PPNS_, _PPNC_);
  
    if (DOOUT) {
      printf ("Output\n");
      getFreqArray(dt,frq);
      for (int i=0; i<_PPNF_; i++)
	printf ("%g %g \n",frq[i]/1e6,pwr[i]);
    }
  
  return 0;
}
