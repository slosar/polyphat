#include "settings.h"
#include "stdint.h"
#include "processor_cuda.h"
#include "generator.h"
#define _XOPEN_SOURCE 700
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "inttypes.h"


int main() {
  printf("Allocating and sampling\n");
  uint8_t *buf1=malloc(BUFFER_SIZE); /*alloc_sample_buffer();*/
  uint8_t *buf2=malloc(BUFFER_SIZE); /*alloc_sample_buffer();*/
  generate_white(buf1,10.,123);
  write_bin (buf1,"white.bin");
  generate_tone(buf2,800.,50.,0.,123);
  write_bin (buf2,"800MHz.bin");
  for (size_t i=0;i<100;i++) {
    printf ("%i %i %i\n", (int)i,buf1[i],buf2[i]);
  }
  return 0;
}
