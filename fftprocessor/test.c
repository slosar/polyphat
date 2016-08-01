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
  print_settings();
  
  printf("Reading files\n");
  uint8_t *buf1=alloc_sample_buffer();
  uint8_t *buf2=alloc_sample_buffer();
  float* freq=alloc_power();
  float* power=alloc_power();
  read_bin(buf1,"800MHz.bin");
  //read_bin(buf2,"100MHz.bin");
  printf ("Launching cuda_test\n");
  cuda_test(buf1,freq,power);
  return 0;
}
