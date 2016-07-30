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
  printf("Reading files\n");
  uint8_t *buf1=malloc(BUFFER_SIZE); /*alloc_sample_buffer();*/
  uint8_t *buf2=malloc(BUFFER_SIZE); /*alloc_sample_buffer();*/
  read_bin(buf1,"white.bin");
  //read_bin(buf2,"100MHz.bin");
  printf ("Launching cuda_test\n");
  cuda_test(buf1);
  return 0;
}
