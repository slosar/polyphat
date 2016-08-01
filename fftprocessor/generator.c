
/* Generates white noise */
#include "generator.h"
#include "settings.h"
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "assert.h"
#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void generate_white (uint8_t *buffer, float amplitude, unsigned int seed) {
  srand(seed);
  for (size_t i=0; i<BUFFER_SIZE; i++) {
    buffer[i]=(uint8_t)(128.0+sampleNormal()*amplitude);
  }
}


void generate_tone (uint8_t *buffer, float amplitude, float freq, float phi, unsigned int seed) {
  srand(seed);
  for (size_t i=0; i<BUFFER_SIZE; i++) {
    buffer[i]=(uint8_t)(128.0+amplitude*sin(phi+2*M_PI*freq*1e6*(i*DELTA_T))+0.5*sampleNormal());
  }
}

void read_bin(uint8_t *buffer, char* fname) {
  FILE *ptr_fp=fopen(fname, "r");
  if (ptr_fp==NULL) {
    printf ("File not found! %s ",fname);
    exit(1);
  }    
  size_t numread=fread(buffer, BUFFER_SIZE, 1, ptr_fp);
  assert(numread==1);
  fclose(ptr_fp);
}

void write_bin(uint8_t *buffer, char* fname) {
  FILE *ptr_fp=fopen(fname, "wb");
  fwrite(buffer, BUFFER_SIZE, 1, ptr_fp);
  fclose(ptr_fp);
}
