#include "settings.h"
#include <stdio.h>

void print_settings() {
  printf ("******************************************\n");
  printf ("FFTPROCESSOR, VERSION: %s\n",VERSION);
  printf ("SAMPLING RATE: %g \n", SAMPLING_RATE);
  printf ("FFT_SIZE: %i \n", FFT_SIZE);
  printf ("NUM_FFT: %i \n", NUM_FFT);
  printf ("BUFFER_SIZE: %i = %g ms\n",BUFFER_SIZE,BUFFER_SIZE*DELTA_T/1e-3);
  printf ("NUMMIN/MAX: %f %f \n",NUMIN,NUMAX);
  printf ("NUM_NUBINS: %i\n", num_nubins());
  printf ("******************************************\n");
  
}

